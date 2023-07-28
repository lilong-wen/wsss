import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Set wsss', add_help=False)

    #training
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_cls_head', default=5e-5, type=float)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=3, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--end_epochs', default=10, type=int)
    parser.add_argument('--num_refines', default=1, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)

    parser.add_argument('--seed', default=75, type=int)
    parser.add_argument('--device', default="cuda", type=str)

    #weight dict
    parser.add_argument('--loss_ctrl_points', default=5.0, type=float)
    parser.add_argument('--loss_texts', default=2.0, type=float)
    parser.add_argument('--loss_bbox', default=5.0, type=float)
    parser.add_argument('--loss_ce', default=2.0, type=float)
    parser.add_argument('--loss_giou', default=2.0, type=float)
    parser.add_argument('--loss_enc_bbox', default=5.0, type=float)
    parser.add_argument('--loss_enc_ce', default=2.0, type=float)
    parser.add_argument('--loss_enc_giou', default=2.0, type=float)
    parser.add_argument('--aux_loss', default=True, type=bool)
    parser.add_argument('--box_jitter', default=0.1, type=float)
    parser.add_argument('--loss_img_cls_r', default=1.0, type=float)
    parser.add_argument('--loss_img_cls_s', default=1.0, type=float)
    parser.add_argument('--loss_img_cls_e', default=1.0, type=float)
    parser.add_argument('--loss_img_clip', default=1.0, type=float)


    #dist
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--dist_url', default='env://', type=str)
    parser.add_argument('--dist_backend', default='nccl', type=str)


    # backbone
    parser.add_argument('--vocab_size', default=96, type=int)
    parser.add_argument('--embed_dim', default=384, type=int)
    parser.add_argument('--image_resolution', default=512, type=int)
    parser.add_argument('--vision_layers', nargs='+', default=[3, 4, 6, 3], type=int)
    parser.add_argument('--vision_width', default=64, type=int)
    parser.add_argument('--context_length', default=25, type=int)
    parser.add_argument('--transformer_width', default=64, type=int)
    parser.add_argument('--transformer_heads', default=8, type=int)
    parser.add_argument('--transformer_layers', default=6, type=int)
    parser.add_argument('--transformer_decoder_layers', default=6, type=int)

    #transformer
    # parser.add_argument('--hidden_dim', default=384, type=int)
    # parser.add_argument('--dropout', default=0.1, type=float)
    # parser.add_argument('--nheads', default=8, type=int)
    # parser.add_argument('--num_queries', default=300, type=int)
    # parser.add_argument('--dim_feedforward', default=192, type=int)
    # parser.add_argument('--enc_layers', default=3, type=int)
    # parser.add_argument('--dec_layers', default=6, type=int)
    # parser.add_argument('--pre_norm', default=False, type=bool)
    # parser.add_argument('--return_intermediate_dec', default=True, type=bool)

    # #Position encodign
    # parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'))


    #DETR
    # parser.add_argument('--num_classes', default=97, type=int)
    # parser.add_argument('--aux_loss', default=True, type=bool)
    # parser.add_argument('--num_refine_steps', default=1, type=int)

    # Deformable DETR
    parser.add_argument('--d_model', default=256, type=int)
    parser.add_argument('--num_classes', default=1, type=int)
    parser.add_argument('--nhead', default=8, type=int)
    parser.add_argument('--num_encoder_layers', default=6, type=int)
    parser.add_argument('--num_decoder_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=1024, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--activation', default='relu', type=str)
    parser.add_argument('--return_intermediate_dec', default=True, type=bool)
    parser.add_argument('--num_feature_levels', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--num_proposals', default=100, type=int)
    parser.add_argument('--pos_embed_scale', default=6.283185307179586, type=float)
    #TODO num of point should be at least 8, but box for now
    parser.add_argument('--num_ctrl_points', default=4, type=int)
    parser.add_argument('--max_text_len', default=25, type=int)
    parser.add_argument('--voc_size', default=96, type=int)
    parser.add_argument('--sigmoid_offset', default=True, type=bool)

    #Matcher
    #TODO old matcher, delete
    parser.add_argument('--set_cost_class', default=2, type=int)
    parser.add_argument('--set_cost_bbox', default=5, type=int)
    parser.add_argument('--set_cost_giou', default=2, type=int)
    parser.add_argument('--hung_match_ratio', default=5, type=int)
    # Matcher
    parser.add_argument('--box_class_weight', default=2.0, type=float)
    parser.add_argument('--box_coord_weight', default=5.0, type=float)
    parser.add_argument('--box_giou_weight', default=2.0, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--focal_gamma', default=2.0, type=float)

    parser.add_argument('--point_class_weight', default=2.0, type=float)
    parser.add_argument('--point_coord_weight', default=5.0, type=float)


    #criterion
    # parser.add_argument('--num_queries', default=300, type=int)
    # parser.add_argument('--focal_alpha', default=0.25, type=float)
    #TODO fact check, if 0.5 if better than 5.0
    # parser.add_argument('--focal_gamma', default=0.5, type=float)

    #attention2box
    parser.add_argument('--cam_thr', default=0.2, type=float)
    parser.add_argument('--multi_box_ratio', default=0.5, type=float)

    #TODO change this parameter, not safe
    #hack parameter, may cause some trouble
    parser.add_argument('--idx_mask', default=95, type=int)


    parser.add_argument('--train_file', nargs='+', default=['./data/SynthText/sample.csv'], type=str)
    parser.add_argument('--test_file', nargs='+', default='totaltext', type=str)
    parser.add_argument('--char_dict_path', default='./data/char_dict', type=str)


    return parser