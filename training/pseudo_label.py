import torch
import cv2
import numpy as np
import copy


def resize_cam(cam, size=(224, 224)):
    cam = cv2.resize(cam , (size[0], size[1]))
    cam = cam - cam.min()
    cam = cam / cam.max()
    return cam

def get_bboxes(cam, cam_thr=0.2):
    """
    cam: single image with shape (h, w, 1)
    thr_val: float value (0~1)
    return estimated bounding box
    """
    cam = (cam * 255.).astype(np.uint8)
    map_thr = cam_thr * np.max(cam)

    _, thr_gray_heatmap = cv2.threshold(cam,
                                        int(map_thr), 255,
                                        cv2.THRESH_TOZERO)
    #thr_gray_heatmap = (thr_gray_heatmap*255.).astype(np.uint8)

    contours, _ = cv2.findContours(thr_gray_heatmap,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        estimated_bbox_list = []
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        estimated_bbox = [x, y, x + w, y + h]
    else:
        estimated_bbox = [0, 0, 1, 1]
        
    estimated_bbox_list.append(estimated_bbox)

    return estimated_bbox_list  #, thr_gray_heatmap, len(contours)


def get_multi_bboxes(cam, cam_thr=0.2, area_ratio=0.5):
    """
    cam: single image with shape (h, w, 1)
    thr_val: float value (0~1)
    return estimated bounding box
    """
    cam = (cam * 255.).astype(np.uint8)
    map_thr = cam_thr * np.max(cam)

    _, thr_gray_heatmap = cv2.threshold(cam,
                                        int(map_thr), 255,
                                        cv2.THRESH_TOZERO)
    #thr_gray_heatmap = (thr_gray_heatmap*255.).astype(np.uint8)

    contours, _ = cv2.findContours(thr_gray_heatmap,
                                       cv2.RETR_TREE,
                                       cv2.CHAIN_APPROX_SIMPLE)
                                       
    if len(contours) != 0:
        estimated_bbox = []
        areas = list(map(cv2.contourArea, contours))
        area_idx = sorted(range(len(areas)), key=areas.__getitem__, reverse=True)
        for idx in area_idx:
            if areas[idx] >= areas[area_idx[0]] * area_ratio:
                c = contours[idx]
                x, y, w, h = cv2.boundingRect(c)
                estimated_bbox.append([x, y, x + w, y + h])
        # areas1 = sorted(areas, reverse=True)
        
        # pdb.set_trace()
        
        # estimated_bbox = [x, y, x + w, y + h]
    else:
        estimated_bbox = [[0, 0, 1, 1]]

    return estimated_bbox  #, thr_gray_heatmap, len(contours)


@torch.no_grad()
def get_pseudo_label_multi_boxes(outputs, samples, targets, args):
    def box_xyxy_to_cxcywh(x):
        x0, y0, x1, y1 = x[...,0], x[...,1], x[...,2], x[...,3]
        b = [(x0 + x1) / 2, (y0 + y1) / 2,
            (x1 - x0), (y1 - y0)]
        return torch.stack(b, dim=-1)

    def normalize_bbox(boxes, image_size):
        h, w = image_size
        boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32).to(boxes.get_device())
        return boxes

    device = samples.tensors.get_device()
    cams = outputs['cam_word']
    max_len = targets[0]['texts_pad'].shape[0]
    Pseudo_labels = []
    for batch_i in range(cams.shape[0]):
        image_size_i = samples.tensors.shape[-2:]
        # image_label_i = targets[batch_i]['text'].data.cpu().numpy().reshape(-1)
        image_label_i = targets[batch_i]['texts_pad'].data.cpu().numpy()
        estimated_bbox = []
        estimated_class= []
        for class_i in range(max_len):
            if image_label_i[class_i].sum() > 0:
                cam_i = cams[batch_i, [class_i], :, :]
                cam_i = torch.mean(cam_i, dim=0, keepdim=True)
                cam_i = cam_i.detach().cpu().numpy().transpose(1, 2, 0)
                cam_i = resize_cam(cam_i, size=image_size_i)
                bbox = get_bboxes(cam_i, cam_thr=args.cam_thr)
                # bbox = get_multi_bboxes(cam_i, cam_thr=args.cam_thr, area_ratio=args.multi_box_ratio)
                bbox = torch.tensor(bbox)
                bbox = box_xyxy_to_cxcywh(bbox)
                estimated_bbox.append(bbox)
                if bbox.shape[0] > 1:
                    for _ in range(bbox.shape[0]):
                        # estimated_class.append(class_i + 1)
                        estimated_class.append(image_label_i[class_i])
                else:
                        estimated_class.append(image_label_i[class_i])

        estimated_bbox = torch.cat(estimated_bbox, dim=0).to(device)
        estimated_label = torch.zeros((estimated_bbox.shape[0], 1), dtype=torch.int64).squeeze(-1).to(device)
        estimated_class= torch.tensor(estimated_class).to(device)

        estimated_bbox = normalize_bbox(estimated_bbox, image_size_i)
        Pseudo_labels.append({'boxes':estimated_bbox, 'texts': estimated_class, 'labels': estimated_label})
    
    return Pseudo_labels


@torch.no_grad()
def output_to_pseudo_label(outputs, targets, postprocessors):
    # device = samples.tensors.get_device()
    orig_target_sizes = torch.stack([t["ori_size"] for t in targets], dim=0)
    pred_results = postprocessors['bbox'](outputs, orig_target_sizes, targets)
    Pseudo_labels= []
    for idx, result in enumerate(pred_results):
        Pseudo_labels.append(copy.deepcopy(targets[idx]))
        det_cls = result['labels'].detach().clone()
        det_box = result['boxes'].detach().clone()
        det_score=result['scores'].detach().clone()
        Pseudo_labels[-1].update({f'labels':det_cls, 
                        f'boxes': det_box, 
                        f'scores': det_score})
    return Pseudo_labels

@torch.no_grad()
def get_refinements_pseudo_label(outputs, targets, postprocessors, args):
    targets_refine = {}
    # targets_refine[1] = output_to_pseudo_label(outputs[0], samples, targets, args, postprocessors)
    for k, v in outputs.items():
        if k == args.num_refines:
            break
        targets_refine[k+1] = output_to_pseudo_label(v,  targets, postprocessors)
    
    return targets_refine