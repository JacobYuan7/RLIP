import argparse

import torch
from torch import nn
import os

def replace_ParSe_context_decoder_with_RLIP():
    ParSe_path = '/PATH/TO/params/detr-r50-pre-hico_SepTransformerv2_query128.pth'
    RLIP_ParSe_path = '/PATH/TO/logs/ParSe_CEFocal_BiasT_Freq500_GIoUVb_PseuEuc03_aftNorm_bs128_500e/checkpoint.pth'
    save_path = '/PATH/TO/params/detr-r50-pre-hico_SepTransformerv2_query128_RLIP-ParSe_253ep.pth'

    ParSe_ps = torch.load(ParSe_path)
    RLIP_ParSe_ps = torch.load(RLIP_ParSe_path)

    for k in list(ParSe_ps['model'].keys()):
        if 'verb_decoder' in k:
            ParSe_ps['model'].pop(k)
    # Ensure there is no verb_decoder
    for k in list(ParSe_ps['model'].keys()):
        if 'verb_decoder' in k:
            print('Error')
    print(len(ParSe_ps['model'].keys()))

    for k in list(RLIP_ParSe_ps['model'].keys()):
        if 'verb_decoder' in k:
            new_k = k
            if 'cross_attn_image' in k:
                new_k = k.replace('cross_attn_image', 'multihead_attn')
            if 'norm3' in k:
                new_k = k.replace('norm3', 'norm2')
            if 'norm4' in k:
                new_k = k.replace('norm4', 'norm3')
            ParSe_ps['model'][new_k] = RLIP_ParSe_ps['model'][k]
            # DABDETR_ps['model'][k.replace('decoder', 'verb_decoder')] = DETR_ps['model'][k].clone()
    print(len(ParSe_ps['model'].keys()))
    
    torch.save(ParSe_ps, save_path)


def create_COCO_init_for_RLIP_ParSe():
    sepv2 = "detr-r50-pre-hico_SepTransformerv2_query200.pth"
    save_path = 'detr-r50-pre-hico_SepTransformerv2_query200_for_RLIP-ParSe.pth'
    # sepv2 = '/PATH/TO/params/detr-r50-pre-hico_SepTransformerv2_query200.pth'
    # save_path = '/PATH/TO/params/detr-r50-pre-hico_SepTransformerv2_query200_for_RLIP-ParSe.pth'
    # sepv2 = '/PATH/TO/params/detr-r101-pre-hico_SepTransformerv2_query200_no_obj_class.pth'
    # save_path = '/PATH/TO/params/detr-r101-pre-hico_SepTransformerv2_query200_no_obj_class_for_RLIP-ParSe.pth'
    sepv2_ps = torch.load(sepv2)
    sepv2_ps_new = torch.load(sepv2)

    for k in list(sepv2_ps['model'].keys()):
        if 'verb_decoder' in k or 'ho_decoder' in k:
            new_k = k
            if 'multihead_attn' in k:
                new_k = k.replace('multihead_attn', 'cross_attn_image')
            if 'norm2' in k:
                new_k = k.replace('norm2', 'norm3')
            if 'norm3' in k:
                new_k = k.replace('norm3', 'norm4')
            sepv2_ps_new['model'][new_k] = sepv2_ps['model'][k]

    torch.save(sepv2_ps_new, save_path)

def create_COCO_init_for_RLIP_ParSeD():
    sepv2 = '/PATH/TO/params/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint_numrefpoints4_SepDDETRHOIv3_converted.pth'
    save_path = '/PATH/TO/params/r50_deformable_detr_COCO_for_RLIP-ParSeD.pth'
    sepv2_ps = torch.load(sepv2)
    sepv2_ps_new = torch.load(sepv2)

    print(sepv2_ps['model'].keys())
    

if __name__=="__main__":
    # replace_ParSe_context_decoder_with_RLIP()
    create_COCO_init_for_RLIP_ParSe()
    # create_COCO_init_for_RLIP_ParSeD()