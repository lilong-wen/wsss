import torch
from .backbone import build_backbone
from .transformer import build_transformer
from .conditional_detr import Conditional_DETR
from .matcher import build_matcher
from .criterion import SetCriterion
from .postprocess import PostProcess



def build_model(args):
    num_classes = args.num_classes
    device = torch.device(args.device)

    backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = Conditional_DETR(
        backbone,
        transformer,
        num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        num_refines=args.num_refine_steps,
    )

    matcher = build_matcher(args)

    weight_dict = {'loss_ce': args.char_loss,
                   'loss_bbox': args.bbox_loss,
                   'clip': args.clip_loss,
                   'match': args.match_loss}

    losses = ['char', 'clip', 'boxes', 'match']

    criterion = SetCriterion(num_classes=args.num_classes,
                            matcher=matcher,
                            weight_dict=weight_dict,
                            focal_alpha=args.focal_alpha,
                            losses=losses,
                            gamma=args.focal_gamma,
                            box_jitter=args.box_jitter)
    criterion.to(device)

    postprocess = {'bbox': PostProcess()}

    return model, criterion, postprocess