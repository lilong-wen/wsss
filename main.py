import random
import numpy as np
import torch
import argparse
import utils.misc as utils
from utils.preprocess import _transform
from models import build_model
from datasets import build_dataset
from training.engine import train_one_epoch
from training.params import get_args_parser

import logging

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main(args):

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.gpu)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    preprocess_fn_train = _transform(args.image_resolution, is_train=True)
    preprocess_fn_val = _transform(args.image_resolution, is_train=False)

    data_train = build_dataset(image_set='train', args=args, \
            preprocess_fn=preprocess_fn_train)
    data_val = build_dataset(image_set='val', args=args, \
            preprocess_fn=preprocess_fn_val)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
            find_unused_parameters=True)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() \
                if "backbone" not in n and p.requires_grad]
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() \
                if "backbone" in n and p.requires_grad and 'blocks_token_only' not in n],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() \
                if "backbone" in n and p.requires_grad and 'blocks_token_only' in n],
            "lr": args.lr_cls_head,
        },

    ]

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)



    start_epoch = 0
    for epoch in range(start_epoch, args.end_epochs):

        if args.distributed:
            data_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(model, criterion, \
                                    data_train.dataloader, optimizer, \
                                    device, epoch, postprocessors, args)

        lr_scheduler.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set wsss', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)