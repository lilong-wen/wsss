import torch
import sys
import math
import copy
import time
import utils.misc as utils
from .pseudo_label import get_pseudo_label_multi_boxes, get_refinements_pseudo_label

def train_one_epoch(model, criterion, data, optimizer, device, eopch, postprocessors, args):
    
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    gettime = lambda: time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime())
    header = f'[{gettime()}] ==> Epoch: [{eopch}]'
    print_freq = 1
    #TODO
    #LOSS DICT here
    weight_dict = copy.deepcopy(criterion.weight_dict)

    for images, texts, targets in metric_logger.log_every(data, print_freq, header):
        # images, texts = images.to(device), [item.to(device) for item in texts]
        images, texts = images.to(device), texts.to(device)
        tmp_vis_sample = images.tensors
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images, texts)
        pseudo_label = get_pseudo_label_multi_boxes(outputs, images, targets, args)

        for t, p in zip(targets, pseudo_label):
            t.update(p)

        #TODO lable refinement add here
        # 

        loss_dict = criterion(outputs, targets)

        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if args.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
                    