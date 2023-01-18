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
from random import choice, uniform
from transformers import RobertaModel, RobertaTokenizerFast, BertTokenizerFast, BertModel
import json
from collections import OrderedDict
import torch

def extract_textual_features(text_encoder_type = "roberta-base"):
    tokenizer = RobertaTokenizerFast.from_pretrained(text_encoder_type)
    text_encoder = RobertaModel.from_pretrained(text_encoder_type)

    with open("/PATH/TO/vg_keep_names_v1_no_lias_freq.json", "r") as f:
        vg_keep_names = json.load(f)
    print('Loading json finished.')
    relationship_names = vg_keep_names["relationship_names"]
    print(relationship_names)
    object_names = vg_keep_names["object_names"]
    print(len(relationship_names), len(object_names))
    if "relationship_freq" in vg_keep_names.keys():
        relationship_freq = vg_keep_names["relationship_freq"]
    if "object_freq" in vg_keep_names.keys():
        object_freq = vg_keep_names["object_freq"]
    
    len_rel = len(relationship_names)
    len_obj = len(object_names)
    len_text = len_rel + len_obj
    flat_text = relationship_names + object_names
    # flat_text = flat_text[-1001:]
    # len_text = 1001
    
    text_feature = []
    flag = 0
    while flag < len_text:
        if flag + 1000 < len_text:
            partial_text = flat_text[flag: (flag+1000)]
            flag += 1000
        else:
            partial_text = flat_text[flag:]
            flag = len_text 
        print(flag)

        flat_tokenized = tokenizer.batch_encode_plus(partial_text, padding="longest", return_tensors="pt")
        # tokenizer: dict_keys(['input_ids', 'attention_mask'])
        #            'input_ids' shape: [text_num, max_token_num]
        #            'attention_mask' shape: [text_num, max_token_num]
        encoded_flat_text = text_encoder(**flat_tokenized)
        text_memory = encoded_flat_text.pooler_output
        text_feature.append(text_memory.detach())
        # text_feature.update({t:m for t, m in zip(partial_text, text_memory)})
    text_feature = torch.cat(text_feature, dim = 0)
    print('Feature extraction finished!')
    # m = text_feature[[0,8,10]]
    # print(torch.einsum('ab,cb->ac',m,m))

    rel_feature = {}
    obj_feature = {}
    for idx, f in enumerate(text_feature):
        if idx < len_rel:
            rel_feature[flat_text[idx]] = f.numpy()
        else:
            obj_feature[flat_text[idx]] = f.numpy()
    print(len(rel_feature), len(obj_feature))
    
    np.savez_compressed('vg_keep_names_v1_no_lias_freq_text_feature.npz',
                        rel_feature = rel_feature,
                        obj_feature = obj_feature)
    
    # test similarity
    # print(rel_feature['on'].T)
    # test_text = [rel_feature['on'], rel_feature['on a'], rel_feature['on top of'], , rel_feature['are on']]

    # text_memory_resized = model.module.transformer.resizer(text_memory)
    # text_memory_resized = text_memory_resized.unsqueeze(dim = 1).repeat(1, args.batch_size, 1)
    # # text_attention_mask = torch.ones(text_memory_resized.shape[:2], device = device).bool()
    # text_attention_mask = torch.zeros(text_memory_resized.shape[:2], device = device).bool()
    # text = (text_attention_mask, text_memory_resized, obj_pred_names_sums)
    # # kwargs = {'text':text}

if __name__=='__main__':
    extract_textual_features(text_encoder_type = "roberta-base")
    

    