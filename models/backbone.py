# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2022) B-
# ytedance Inc..  
# *************************************************************************


from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
from utils.misc import NestedTensor
from .position_encoding import build_position_encoding
from adet.layers.pos_encoding import PositionalEncoding2D

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        # self.positional_embedding = PositionEmbeddingSine(spacial_dim ** 2 + 1, num_pos_feats=embed_dim // 2)
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        # self.positional_embedding = PositionEmbeddingSine(spacial_dim ** 2, num_pos_feats=1024)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, att_maps = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=True
        )

        return x, att_maps


class ResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x_1 = self.layer2(x)
        x_2 = self.layer3(x_1)
        x_3 = self.layer4(x_2)
        x, att_maps = self.attnpool(x_3)

        return x, att_maps, [x_1, x_2, x_3]


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x, mask):
        mask = mask.to(device=x.device) if mask is not None else None
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, key_padding_mask=mask, attn_mask=self.attn_mask)[0]
        
        # return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]


    def forward(self, x: list):
        x, mask = x
        x = x + self.attention(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return [x, mask]


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x):
        return self.resblocks(x)


class ResidualAttentionBlockDecoder(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, q, k, v, im_m):
        self.attn_mask = self.attn_mask.to(dtype=q.dtype, device=q.device) if self.attn_mask is not None else None
        return self.attn(q, k, v, attn_mask=self.attn_mask, key_padding_mask=im_m)
        
        # return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]


    def forward(self, x: list):
        if len(x) == 4:
            q, k, v, im_m = x
        else:
            q, k, v, im_m, m = x
        
        q_, m = self.attention(q, k, v, im_m)
        q = q + self.ln_1(q_)
        q = q + self.mlp(self.ln_2(q))
        return [q, k, v, im_m, m]


class TransformerDecoder(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlockDecoder(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x):
        return self.resblocks(x)



class oCLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 transformer_decoder_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        #TODO change this into config
        self.idx_masks = 95

        vision_heads = vision_width * 32 // 64

        self.visual = ResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
        )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.transformer_heads = transformer_heads
        self.transformer_width = transformer_width
    
        self.transformer_decoder = TransformerDecoder(
            width=embed_dim,
            layers=transformer_decoder_layers,
            heads=transformer_heads,
        )

        self.vocab_size = vocab_size + 1
        self.token_embedding = nn.Embedding(self.vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
    
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.ln_final_decoder = LayerNorm(embed_dim)
        self.text_class = nn.Linear(embed_dim, self.vocab_size)
        self.image_pos = nn.Parameter(torch.randn((image_resolution // 32) ** 2, embed_dim) / embed_dim ** 0.5)

        self.initialize_parameters()
        

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        if isinstance(self.visual, ResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)


        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        for block in self.transformer_decoder.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
        nn.init.normal_(self.text_class.weight, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal

        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype
        # return self.visual.backbone.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text, mask):
        # text: [batch_size, n_ctx] 64 x 77
        batch_size, n_words, n_chars = text.shape
        text = text.reshape(batch_size * n_words, n_chars)
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        # x: [batch_size, n_ctx, d_model] 64 x 77 x 512

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer([x, mask])[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x: [batch_size, n_ctx, d_model] 64 x 77 x 512

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        x = x.reshape(batch_size, n_words, x.shape[-1])
        # x: [batch_size, d_out] 64 x 1024

        return x

    def att_text_to_image(self, encoded_image, encoded_text, image_mask):
        x = encoded_text.permute(1, 0, 2)  # NLD -> LND
        tmp = self.transformer_decoder([x, encoded_image + self.image_pos[:, None, :].to(encoded_image.dtype), encoded_image, image_mask])
        x = tmp[0]
        m = tmp[4]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final_decoder(x).type(self.dtype)
        return x, m

    #TODO modify later
    # def att_text_line_to_image(self, encoded_image, encoded_text, image_mask):
    #     x = encoded_text.permute(1, 0, 2)  # NLD -> LND
    #     tmp = self.transformer_decoder([x, encoded_image + self.image_pos[:, None, :].to(encoded_image.dtype), encoded_image, image_mask])
    #     x = tmp[0]
    #     m = tmp[4]
    #     x = x.permute(1, 0, 2)  # LND -> NLD
    #     x = self.ln_final_decoder(x).type(self.dtype)
    #     return x, m

    def forward(self, tensor_list, text):

        image, image_mask = tensor_list.decompose()
        # tmp_vis_image_mask = image_mask[0].reshape(16, 61)
        m_c, m_h, m_w = image_mask.shape
        image_mask = image_mask.flatten(1, 2)

        encoded_image, att_maps, multi_features = self.encode_image(image)
        encoded_texts = self.encode_text(text, None)
        logit_scale = self.logit_scale.exp()

        image_features = encoded_image[0]
        encoded_image = encoded_image[1:]

        text_features = torch.mean(encoded_texts, dim=1)

        text_image_enc, char_mask = self.att_text_to_image(encoded_image.repeat(1,3,1), encoded_texts, image_mask.repeat(3,1))
        char_mask_r, char_mask_s, char_mask_e = torch.split(char_mask, m_c)
        char_mask_w = torch.max(torch.max(char_mask_r, char_mask_s), char_mask_e)

        # text_image_enc_r, text_image_enc_s, text_image_e = torch.split(text_image_enc, m_c)
        text_logits = self.text_class(text_image_enc)

        # if self.training:
            # return image_features, text_features, text_logits, logit_scale
        # else:
            # return image_features, text_features, text_logits, att_maps, char_mask, logit_scale    

        return {'x_logits': text_logits, 
                'logit_scale': logit_scale,
                'cam_cls': char_mask_r.unflatten(-1, (m_h, m_w)),
                'cam_word': char_mask_w.unflatten(-1, (m_h, m_w)),
                'img_feature': multi_features, 
                'img_feature_s': image_features,
                'text_features': text_features}


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
        # for name in ["proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        #TODO modify to avoid hard coding
        self.feature_strides = [8, 16, 32]
        self.num_channel = 2048

    def forward(self, tensor_list: NestedTensor, texts):

        backbone_out = self[0](tensor_list, texts)
        tmp_vis_tensor = tensor_list.tensors
        features = backbone_out['img_feature']
        masks = self.mask_out_padding(
            [features_per_level.shape for features_per_level in features],
            tensor_list.tensors.shape[-2:],
            tensor_list.tensors.device,
        )
        assert len(features) == len(masks)

        out = []
        pos = []
        for i, x in enumerate(features):
            out.append(NestedTensor(x, masks[i]))
            # position encoding
            pos.append(self[1](out[i]).to(x.dtype))

        backbone_out['img_feature'] = out

        return backbone_out, pos        
        # fea_size = tensor_list.mask.shape[-1]
        # x = x.permute(1,2,0).unflatten(2, (fea_size, fea_size))
        # m = tensor_list.mask
        # assert m is not None
        # mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        # det_x = NestedTensor(x, mask)
        # backbone_out['img_feature'] = det_x
        # pos = []
        # # position encoding
        # pos.append(self[1](det_x).to(det_x.tensors.dtype))

        # return backbone_out, pos

    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides)
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones((N, H, W), dtype=torch.bool, device=device)
            # for img_idx, (h, w) in enumerate(image_sizes):
                # masks_per_feature_level[
                    # img_idx,
                    # : int(np.ceil(float(h) / self.feature_strides[idx])),
                    # : int(np.ceil(float(w) / self.feature_strides[idx])),
                # ] = 0
            masks.append(masks_per_feature_level)
        return masks

def build_backbone(args):

    backbone = oCLIP(
        args.embed_dim,
        args.image_resolution, 
        args.vision_layers, 
        args.vision_width, 
        args.context_length, 
        args.vocab_size, 
        args.transformer_width, 
        args.transformer_heads, 
        args.transformer_layers,
        args.transformer_decoder_layers,
    )

    # position_embedding = build_position_encoding(args)
    model = Joiner(backbone, PositionalEncoding2D(args.d_model//2, normalize=True))

    return model
