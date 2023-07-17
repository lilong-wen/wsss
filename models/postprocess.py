import torch.nn as nn
import torch
import sys
sys.path.append("..")
from utils import box_ops



class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes, targets=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        device = out_logits.get_device()

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        prob = out_logits.sigmoid()

        top_values, top_indexes = torch.max(prob, dim=1)
        top_boxes = torch.gather(out_bbox, 1, top_indexes.unsqueeze(-1).repeat(1, 1, 4))
        top_labels = torch.arange(out_logits.shape[2]).unsqueeze(0).repeat(out_logits.shape[0], 1)
        scores, boxes, labels = [], [], []
        for ii in range(len(targets)):
            tmp_labels, tmp_scores, tmp_boxes = [], [], []
            for cc in range(out_logits.shape[2]):
                if cc in targets[ii]['labels']:
                    tmp_labels.append(cc)
                    tmp_scores.append(top_values[ii][cc].reshape(-1))
                    tmp_boxes.append(top_boxes[ii][cc].reshape(1, -1))
            labels.append(torch.tensor(tmp_labels).to(device))
            scores.append(torch.cat(tmp_scores, dim=0))
            boxes.append(torch.cat(tmp_boxes, dim=0))

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results
