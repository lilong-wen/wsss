import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from utils.misc import NestedTensor,nested_tensor_from_tensor_list, inverse_sigmoid
sys.path.append("../")
import math

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class Conditional_DETR(nn.Module):
    """ This is the Conditional DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False, num_refines=1, drloc=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.num_refines = num_refines
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(num_refines + 1)])
        self.bbox_embed = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(num_refines + 1)])
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.queries_embed_refine = nn.ModuleList([nn.Embedding(num_queries, hidden_dim) for _ in range(num_refines)])
        self.backbone = backbone
        self.aux_loss = aux_loss

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for class_embed in self.class_embed:
            class_embed.bias.data = torch.ones(num_classes) * bias_value

        # init bbox_mebed
        for bbox_embed in self.bbox_embed:
            nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)

    def forward(self, samples: NestedTensor, texts):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        _, _, H, W = samples.decompose()[0].size()
        features, pos = self.backbone(samples, texts)
        src, mask = features['img_feature'].decompose()
        assert mask is not None
        Hs, references = self.transformer(src, mask, self.query_embed.weight, pos[-1],
                                        queries_embed_refine=self.queries_embed_refine)

        references_before_sigmoid = [inverse_sigmoid(r) for r in references]
        # reference_before_sigmoid = inverse_sigmoid(reference)
        out = {}
        for refine_idx in range(self.num_refines + 1):
            bbox_embed = self.bbox_embed[refine_idx]
            class_embed = self.class_embed[refine_idx]
            hs = Hs[refine_idx]
            reference_before_sigmoid = references_before_sigmoid[refine_idx]
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                tmp = bbox_embed(hs[lvl])
                tmp[..., :2] += reference_before_sigmoid
                outputs_coord = tmp.sigmoid()
                outputs_coords.append(outputs_coord)
            outputs_coord = torch.stack(outputs_coords)

            outputs_class = class_embed(hs)
            out_refine = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1], **features}
            if self.aux_loss:
                out_refine['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
            
            out[refine_idx] = out_refine
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
