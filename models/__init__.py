import torch
from .backbone import build_backbone
# from .transformer_conditional import build_transformer
# from .conditional_detr import Conditional_DETR
from .deformable_detr import WSSS
from .matcher import build_matcher
from .criterion import SetCriterion
from .criterion import SetCriterion_with_text
from .postprocess import PostProcess



def build_model(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)
    # transformer = build_transformer(args)

    # model = Conditional_DETR(
        # backbone,
        # transformer,
        # num_classes,
        # num_queries=args.num_queries,
        # aux_loss=args.aux_loss,
        # num_refines=args.num_refine_steps,
    # )
    model = WSSS(
        backbone,
        args
    )

    box_matcher, point_matcher = build_matcher(args)

    #TODO change keys
    # weight_dict = {'loss_ce': args.loss_ce,
    #                # 'loss_ctrl_points': args.loss_ctrl_points,
    #                'loss_texts': args.loss_texts,
    #                }
    weight_dict = {}
    img_weight_dict = {
        "loss_img_cls": args.loss_img_cls_r,
        "loss_img_clip": args.loss_img_clip,
    }
    
    dec_weight_dict = {
        'loss_bbox': args.loss_bbox,
        'loss_giou': args.loss_giou,
        'loss_texts': args.loss_texts,
        'loss_labels': args.loss_ce
    }
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.num_decoder_layers - 1):
            aux_weight_dict.update(
                {k + f'_{i}': v for k, v in dec_weight_dict.items()}
            )
        aux_weight_dict.update(
            # {k + f'_dec': v for k,v in dec_weight_dict.items()}
            {k: v for k,v in dec_weight_dict.items()}
        )
        weight_dict.update(aux_weight_dict)
        weight_dict.update(img_weight_dict)

    #TODO change the loss
    losses = ['char', 'clip', 'boxes', 'match']
    enc_losses = ['labels', 'boxes']
    # dec_losses = ['labels', 'ctrl_points', 'texts']
    dec_losses = ['labels', 'boxes', 'texts']
    img_losses = ['img_cls', 'img_clip']


    # criterion = SetCriterion(num_classes=args.num_classes,
                            # matcher=matcher,
                            # weight_dict=weight_dict,
                            # focal_alpha=args.focal_alpha,
                            # losses=losses,
                            # gamma=args.focal_gamma,
                            # box_jitter=args.box_jitter)
    criterion = SetCriterion_with_text(
        num_classes=1,
        enc_matcher=box_matcher,
        dec_matcher=point_matcher,
        weight_dict=weight_dict,
        enc_losses=img_losses, 
        dec_losses=dec_losses, 
        num_ctrl_points=args.num_ctrl_points, 
        box_jitter=args.box_jitter,
        focal_alpha=0.25, 
        focal_gamma=2.0

    )
    criterion.to(device)

    postprocess = {'bbox': PostProcess()}

    return model, criterion, postprocess