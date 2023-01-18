# ------------------------------------------------------------------------
# RLIP: Relational Language-Image Pre-training
# Copyright (c) Alibaba Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import math
import os
import sys
from typing import Iterable
import numpy as np
import copy
import itertools

import torch

import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.hico_eval import HICOEvaluator
from datasets.vcoco_eval import VCOCOEvaluator
import torch.nn.functional as F

def cal_uniformity_alignment(relation_feature_path):
    verb_class_dict = np.load(relation_feature_path, allow_pickle = True)['verb_class_dict'].item()
    verb_class_tensot_dict = {}
    for verb_idx, ft_list in verb_class_dict.items():
        tensor_list = torch.stack([torch.from_numpy(ft) for ft in ft_list])
        # print(tensor_list.shape)
        verb_class_tensot_dict[verb_idx] = F.normalize(tensor_list, p=2, dim=-1)
    
    print(cal_uniformity(verb_class_tensot_dict), cal_alignment(verb_class_tensot_dict))

def cal_uniformity(feature_dict, t = 2):
    all_feature = torch.cat([j for i,j in feature_dict.items()], dim = 0)
    sq_dist = torch.pdist(all_feature, p = 2).pow(2)
    return sq_dist.mul(-t).exp().mean().log().item()

def cal_alignment(feature_dict, alpha = 2):
    all_feature_num = sum([j.shape[0] for i,j in feature_dict.items()])
    all_align = []

    # V1: Treat features as a sample
    # for i, j in feature_dict.items():
    #     all_align.append(torch.pdist(j, p = 2).pow(alpha).flatten())
    #     print(all_align[-1].shape)
    # return torch.cat(all_align).mean()
    # V2: Treat one class as a sample
    for i, j in feature_dict.items():
        all_align.append(torch.pdist(j, p = 2).pow(alpha).mean().item())
        print(all_align[-1])
    return sum(all_align)/len(all_align)

if __name__ == "__main__":
    relation_feature_path = 'LSE_RQL_RPL_relation_feature.npz'
    cal_uniformity_alignment(relation_feature_path)
