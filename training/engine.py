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

    for images, texts, targets in metric_logger.log_every(data, print_freq, header):
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