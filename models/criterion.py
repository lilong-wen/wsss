import torch.nn as nn
import torch
import sys
from adet.utils.misc import box_cxcywh_to_xyxy, generalized_box_iou
sys.path.append('..')
from utils.misc import accuracy, nested_tensor_from_tensor_list, interpolate, \
        is_dist_avail_and_initialized, get_world_size
    
from utils import box_ops
import copy

import torch.nn.functional as F

class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses, gamma, box_jitter):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.gamma = gamma
        self.eos_coef = 0.1
        self.hung_match_ratio = getattr(matcher, 'match_ratio', 1)
        self.box_jitter = box_jitter
        empty_weight = torch.ones(self.num_classes)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def update_hung_match_ratio(self, ratio=5):
        assert hasattr(self.matcher, 'match_ratio')
        self.matcher.match_ratio = ratio
        self.hung_match_ratio = ratio

    def loss_img_label(self, outputs, targets, *args, **kwargs):
        """Multi-Label Image Classification loss"""
        assert 'x_logits' in outputs and 'x_cls_logits' in outputs
        logits = outputs['x_logits']
        tokens_logits = outputs['x_cls_logits']
        target_class = torch.stack([t["img_label"] for t in targets]).to(logits.get_device()).float()
        loss_label = F.binary_cross_entropy_with_logits(logits, target_class)
        loss_label_tokens = F.binary_cross_entropy_with_logits(tokens_logits, target_class)
        losses = {"img_label_logits": loss_label, "img_label_logits_tokens": loss_label_tokens}

        return losses

    def loss_img_label(self, outputs, targets, *args, **kwargs):
        """Multi-Label Image Classification loss"""
        assert 'x_logits' in outputs and 'x_cls_logits' in outputs
        logits = outputs['x_logits']
        tokens_logits = outputs['x_cls_logits']
        target_class = torch.stack([t["img_label"] for t in targets]).to(logits.get_device()).float()
        loss_label = F.binary_cross_entropy_with_logits(logits, target_class)
        loss_label_tokens = F.binary_cross_entropy_with_logits(tokens_logits, target_class)
        losses = {"img_label_logits": loss_label, "img_label_logits_tokens": loss_label_tokens}

        return losses

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        target_classes_weight = torch.ones_like(target_classes_onehot, device=src_logits.device)


        loss_ce = self.weighted_sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes,
                                                   weights=target_classes_weight, alpha=self.focal_alpha,
                                                   gamma=self.gamma) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_labels_ce(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes - 1,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_img_labels(self, outputs, targets):
        assert 'x_logits_r' in outputs and 'x_logits_s' in outputs and 'x_logits_e' in outputs
        assert 'chars' in targets
        # b_c, _, _ = outputs['x_logits'].shape
        # char_logit_r, char_logit_s, char_logit_e = torch.split(outputs['x_logits'], b_c)
        masked_chars = torch.stack([item['chars'] for item in targets])
        # masked_chars_r, masked_chars_s, masked_chars_e = torch.split(masked_chars, b_c)

        indices = torch.where(masked_chars != 0)
        char_gts = masked_chars[indices]
        chars_logit = outputs['x_logits'][indices]

        loss_char = F.cross_entropy(chars_logit, char_gts)

        losses = {"loss_char": loss_char}

        return losses

    def loss_img_clip(self, outputs, targets):
        
        assert 'img_feature' in outputs and 'text_feature' in outputs

        b_c, _, _ = outputs['x_logits'].shape
        text_feature_r, text_feature_s, text_feature_e = torch.split(outputs['text_features'], b_c)
        # assign the average value of text_feature_r, text_feature_s, text_feature_e to text_features
        text_features = (text_feature_r + text_feature_s + text_feature_e) / 3

        logits_per_image = outputs['logit_scale'] * outputs['img_feature_s'] @ text_features.t()
        logits_per_text = outputs['logit_scale'] * text_features.t() @ outputs['img_feature_s']


        ground_truth = torch.arange(len(logits_per_image)).long()
        
        loss_clip = (F.cross_entropy(logits_per_image, ground_truth) + \
                    F.cross_entropy(logits_per_text, ground_truth)) /2
        
        losses = {"loss_clip": loss_clip}

        return losses
        

    def loss_drloc(self, outputs, *args, **kwargs):
        # try:
        #     print(outputs['drloc_pred'])
        # except BaseException:
        #     pdb.set_trace()
        dr_loc_loss = cal_selfsupervised_loss(outputs['drloc_pred'])
        losses = {"loss_drloc": dr_loc_loss}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'labels_ce': self.loss_labels_ce,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            # 'image_label': self.loss_img_label,
            'img_cls_r': self.loss_img_label,
            'img_cls_s': self.loss_img_label,
            'img_cls_e': self.loss_img_label,
            'img_clip': self.loss_img_clip,
            'drloc': self.loss_drloc,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def get_jittered_box(self, box, box_jitter_x, box_jitter_y, box_jitter, cnt_jitter):
        offset_cx = torch.empty((1000, 1), dtype=box.dtype, device=box.device).uniform_(box_jitter_x[0], box_jitter_x[1]) * box[0, 2]
        offset_cy = torch.empty((1000, 1), dtype=box.dtype, device=box.device).uniform_(box_jitter_y[0], box_jitter_y[1]) * box[0, 3]
        offset_w = torch.empty((1000, 1), dtype=box.dtype, device=box.device).uniform_(-box_jitter, box_jitter) * box[0, 2]
        offset_h = torch.empty((1000, 1), dtype=box.dtype, device=box.device).uniform_(-box_jitter, box_jitter) * box[0, 3]
        offset = torch.cat([offset_cx, offset_cy, offset_w, offset_h], dim=1)
        offset_box = box + offset
        iou, _ = box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(offset_box), box_ops.box_cxcywh_to_xyxy(box))
        keep_idx = torch.where(iou.reshape(-1) > 0.7)[0]
        min_keep_cnt = cnt_jitter if cnt_jitter < keep_idx.numel() else keep_idx.numel()
        box_repeat = box.repeat(cnt_jitter, 1)
        box_repeat[:min_keep_cnt] = offset_box[keep_idx[:min_keep_cnt]]
        return box_repeat


    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        targets_cp = copy.deepcopy(targets)
        if self.training:
            for i in range(len(targets_cp)):
                boxes_repeat = []
                for j in range(len(targets_cp[i]['labels'])):
                    box_j = targets_cp[i]['boxes'][j].reshape(1,4)
                    scale_cx = torch.empty((1000, 1),dtype=box_j.dtype,device=box_j.device).uniform_(1-self.box_jitter, 1+self.box_jitter)
                    scale_cy = torch.empty((1000, 1), dtype=box_j.dtype, device=box_j.device).uniform_(1-self.box_jitter, 1+self.box_jitter)
                    scale_w = torch.empty((1000, 1), dtype=box_j.dtype, device=box_j.device).uniform_(1-self.box_jitter, 1+self.box_jitter)
                    scale_h = torch.empty((1000, 1), dtype=box_j.dtype, device=box_j.device).uniform_(1-self.box_jitter, 1+self.box_jitter)
                    scale = torch.cat([scale_cx, scale_cy, scale_w, scale_h], dim=1)
                    scale_box_j = scale * box_j
                    iou_j, _ = box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(scale_box_j), box_ops.box_cxcywh_to_xyxy(box_j))
                    keep_idx = torch.where(iou_j.reshape(-1) > 0.7)[0]
                    min_keep_cnt = self.hung_match_ratio-1 if (self.hung_match_ratio-1) < keep_idx.numel() else keep_idx.numel()
                    box_j_repeat = box_j.repeat(self.hung_match_ratio,1)
                    box_j_repeat[:min_keep_cnt] = scale_box_j[keep_idx[:min_keep_cnt]]
                    
                    boxes_repeat.append(box_j_repeat)
                targets_cp[i]['boxes'] = torch.cat(boxes_repeat)
                targets_cp[i]['labels'] = targets_cp[i]['labels'].unsqueeze(dim=1).repeat(1,self.hung_match_ratio).reshape(-1,)
                if 'scores' in targets_cp[i]:
                    targets_cp[i]['scores'] = targets_cp[i]['scores'].unsqueeze(dim=1).repeat(1,self.hung_match_ratio).reshape(-1,)

        indices = self.matcher(outputs_without_aux, targets_cp)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets_cp)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets_cp, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets_cp)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    if loss == 'image_label':
                        continue
                    if loss == 'drloc':
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets_cp, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    def weighted_sigmoid_focal_loss(self, inputs, targets, num_boxes, weights, alpha: float = 0.25, gamma: float = 2):
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = -1 (no weighting).
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
        Returns:
            Loss tensor
        """
        EPS = 1e-5
        prob = inputs.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = prob * targets + (1 - prob) * (1 - targets)
        p_t = p_t.clamp(EPS, 1 - EPS)
        loss = weights * ce_loss * ((1 - p_t) ** gamma)
        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        return loss.mean(1).sum() / num_boxes


def relative_constraint_l1(deltaxy, predxy):
    return F.l1_loss(deltaxy, predxy)


def cal_selfsupervised_loss(outs, lambda_drloc=0.1):
    loss = 0.0
    reld_criterion = relative_constraint_l1 

    loss_drloc = 0.0
    for deltaxy, drloc, plane_size in zip(outs.deltaxy, outs.drloc, outs.plz):
        loss_drloc += reld_criterion(deltaxy, drloc) * lambda_drloc
    loss += loss_drloc
    return loss

class SetCriterion_with_text(nn.Module):
    """ This class computes the loss for TESTR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, enc_matcher, dec_matcher, weight_dict, enc_losses, \
                dec_losses, num_ctrl_points, box_jitter, focal_alpha=0.25, focal_gamma=2.0):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        # self.enc_matcher = enc_matcher
        self.dec_matcher = dec_matcher
        self.weight_dict = weight_dict
        self.enc_losses = enc_losses
        self.dec_losses = dec_losses
        self.focal_alpha = focal_alpha
        self.box_jitter = box_jitter
        self.focal_gamma = focal_gamma
        self.num_ctrl_points = num_ctrl_points
        #TODO change hung_match_ratio 
        self.hung_match_ratio = 5

    def loss_labels(self, outputs, targets, indices, num_inst, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)

        target_classes = torch.full(src_logits.shape[:-1], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes_o = torch.cat([t["labels"][J]
                                      for t, (_, J) in zip(targets, indices)])
        if len(target_classes_o.shape) < len(target_classes[idx].shape):
            target_classes_o = target_classes_o[..., None]
        target_classes[idx] = target_classes_o

        shape = list(src_logits.shape)
        shape[-1] += 1
        target_classes_onehot = torch.zeros(shape,
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(-1, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[..., :-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_inst,
                                     alpha=self.focal_alpha, gamma=self.focal_gamma) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - \
                accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_inst):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device)
        card_pred = (pred_logits.mean(-2).argmax(-1) == 0).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_inst):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx].squeeze(-1)
        #src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i]
                                  for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_inst

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_inst
        return losses

    def loss_texts(self, outputs, targets, indices, num_inst):
        assert 'pred_texts' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_texts = outputs['pred_texts'][idx]
        target_texts = torch.cat([t['texts'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        return {'loss_texts': F.cross_entropy(src_texts.transpose(1, 2), target_texts.long())}

    def loss_img_labels(self, outputs, targets, indices_a=None, num_inst=None):
        assert 'x_logits' in outputs
        assert 'chars' in targets[0]
        # b_c, _, _ = outputs['x_logits'].shape
        # char_logit_r, char_logit_s, char_logit_e = torch.split(outputs['x_logits'], b_c)
        masked_chars = torch.vstack([item['chars'] for item in targets])
        # masked_chars_r, masked_chars_s, masked_chars_e = torch.split(masked_chars, b_c)

        indices = torch.where(masked_chars != 0)
        char_gts = masked_chars[indices]
        chars_logit = outputs['x_logits'][indices]

        loss_char = F.cross_entropy(chars_logit, char_gts)

        losses = {"loss_char": loss_char}

        return losses

    def loss_img_clip(self, outputs, targets, indices, num_inst, **kwargs):
        
        assert 'img_feature_s' in outputs and 'text_features' in outputs

        b_c, _ = outputs['img_feature_s'].shape
        text_feature_r, text_feature_s, text_feature_e = torch.split(outputs['text_features'], b_c)
        # assign the average value of text_feature_r, text_feature_s, text_feature_e to text_features
        text_features = (text_feature_r + text_feature_s + text_feature_e) / 3

        logits_per_image = outputs['logit_scale'] * outputs['img_feature_s'] @ text_features.t()
        logits_per_text = outputs['logit_scale'] * text_features @ outputs['img_feature_s'].t()


        ground_truth = torch.arange(len(logits_per_image)).long().to(logits_per_image.device)
        
        loss_clip = (F.cross_entropy(logits_per_image, ground_truth) + \
                    F.cross_entropy(logits_per_text, ground_truth)) /2
        
        losses = {"loss_clip": loss_clip}

        return losses

    def loss_ctrl_points(self, outputs, targets, indices, num_inst):
        """Compute the losses related to the keypoint coordinates, the L1 regression loss
        """
        assert 'pred_ctrl_points' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_ctrl_points = outputs['pred_boxes'][idx]
        target_ctrl_points = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_ctrl_points = F.l1_loss(src_ctrl_points, target_ctrl_points, reduction='sum')

        losses = {'loss_ctrl_points': loss_ctrl_points / num_inst}
        return losses

    def get_jittered_box(self, box, box_jitter_x, box_jitter_y, box_jitter, cnt_jitter):
        offset_cx = torch.empty((1000, 1), dtype=box.dtype, device=box.device).uniform_(box_jitter_x[0], box_jitter_x[1]) * box[0, 2]
        offset_cy = torch.empty((1000, 1), dtype=box.dtype, device=box.device).uniform_(box_jitter_y[0], box_jitter_y[1]) * box[0, 3]
        offset_w = torch.empty((1000, 1), dtype=box.dtype, device=box.device).uniform_(-box_jitter, box_jitter) * box[0, 2]
        offset_h = torch.empty((1000, 1), dtype=box.dtype, device=box.device).uniform_(-box_jitter, box_jitter) * box[0, 3]
        offset = torch.cat([offset_cx, offset_cy, offset_w, offset_h], dim=1)
        offset_box = box + offset
        iou, _ = box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(offset_box), box_ops.box_cxcywh_to_xyxy(box))
        keep_idx = torch.where(iou.reshape(-1) > 0.7)[0]
        min_keep_cnt = cnt_jitter if cnt_jitter < keep_idx.numel() else keep_idx.numel()
        box_repeat = box.repeat(cnt_jitter, 1)
        box_repeat[:min_keep_cnt] = offset_box[keep_idx[:min_keep_cnt]]
        return box_repeat

    @staticmethod
    def _get_src_permutation_idx(indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i)
                               for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i)
                               for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_inst, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'ctrl_points': self.loss_ctrl_points,
            'boxes': self.loss_boxes,
            'texts': self.loss_texts,
            'img_cls': self.loss_img_labels,
            'img_clip': self.loss_img_clip,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_inst, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}
        
        # Retrieve the matching between the outputs of the last layer and the targets
        targets_cp = copy.deepcopy(targets)
        if self.training:
            for i in range(len(targets_cp)):
                boxes_repeat = []
                for j in range(len(targets_cp[i]['labels'])):
                    box_j = targets_cp[i]['boxes'][j].reshape(1,4)
                    scale_cx = torch.empty((1000, 1),dtype=box_j.dtype,device=box_j.device).uniform_(1-self.box_jitter, 1+self.box_jitter)
                    scale_cy = torch.empty((1000, 1), dtype=box_j.dtype, device=box_j.device).uniform_(1-self.box_jitter, 1+self.box_jitter)
                    scale_w = torch.empty((1000, 1), dtype=box_j.dtype, device=box_j.device).uniform_(1-self.box_jitter, 1+self.box_jitter)
                    scale_h = torch.empty((1000, 1), dtype=box_j.dtype, device=box_j.device).uniform_(1-self.box_jitter, 1+self.box_jitter)
                    scale = torch.cat([scale_cx, scale_cy, scale_w, scale_h], dim=1)
                    scale_box_j = scale * box_j
                    iou_j, _ = box_ops.box_iou(box_ops.box_cxcywh_to_xyxy(scale_box_j), box_ops.box_cxcywh_to_xyxy(box_j))
                    keep_idx = torch.where(iou_j.reshape(-1) > 0.7)[0]
                    min_keep_cnt = self.hung_match_ratio-1 if (self.hung_match_ratio-1) < keep_idx.numel() else keep_idx.numel()
                    box_j_repeat = box_j.repeat(self.hung_match_ratio,1)
                    box_j_repeat[:min_keep_cnt] = scale_box_j[keep_idx[:min_keep_cnt]]
                    
                    boxes_repeat.append(box_j_repeat)
                targets_cp[i]['boxes'] = torch.cat(boxes_repeat)
                targets_cp[i]['labels'] = targets_cp[i]['labels'].unsqueeze(dim=1).repeat(1,self.hung_match_ratio).reshape(-1,)
                if 'scores' in targets_cp[i]:
                    targets_cp[i]['scores'] = targets_cp[i]['scores'].unsqueeze(dim=1).repeat(1,self.hung_match_ratio).reshape(-1,)
                if 'texts' in targets_cp[i]:
                    targets_cp[i]['texts'] = targets_cp[i]['texts'].unsqueeze(dim=1).repeat(1,self.hung_match_ratio, 1).flatten(0,1)

        #TODO change matcher to partial match
        indices = self.dec_matcher(outputs_without_aux, targets_cp)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_inst = sum(len(t['labels']) for t in targets_cp)
        num_inst = torch.as_tensor(
            [num_inst], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_inst)
        num_inst = torch.clamp(num_inst / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.dec_losses:
            kwargs = {}
            losses.update(self.get_loss(loss, outputs, targets_cp,
                                        indices, num_inst, **kwargs))
        for loss_i in self.enc_losses:
            losses.update(self.get_loss(loss_i, outputs, targets_cp,
                                        indices, num_inst, **kwargs))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.dec_matcher(aux_outputs, targets_cp)
                for loss in self.dec_losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs['log'] = False
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets_cp, indices, num_inst, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        #TODO enc_outputs may not working, modify the encoder output later
        # if 'enc_outputs' in outputs:
        #     enc_outputs = outputs['enc_outputs']
        #     indices = self.enc_matcher(enc_outputs, targets)
        #     for loss in self.enc_losses:
        #         kwargs = {}
        #         if loss == 'labels':
        #             kwargs['log'] = False
        #         l_dict = self.get_loss(
        #             loss, enc_outputs, targets, indices, num_inst, **kwargs)
        #         l_dict = {k + f'_enc': v for k, v in l_dict.items()}
        #         losses.update(l_dict)

        return losses


def sigmoid_focal_loss(inputs, targets, num_inst, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if loss.ndim == 4:
        return loss.mean((1, 2)).sum() / num_inst
    elif loss.ndim == 3:
        return loss.mean(1).sum() / num_inst
    else:
        raise NotImplementedError(f"Unsupported dim {loss.ndim}")