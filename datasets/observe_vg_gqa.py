# ------------------------------------------------------------------------
# RLIP: Relational Language-Image Pre-training
# Copyright (c) Alibaba Group. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import json 
from pathlib import Path
import torch
import re

def clean_name(name):
    name = re.sub(r"\(.*\)", "", name)
    name = re.sub(r"_", " ", name)
    name = re.sub(r"  ", " ", name)
    return name

def observe_lvis():
    with open(Path('/PATH/TO/data/mdetr_annotations') / "lvis_v1_minival.json", "r") as f:
        lvis_val = json.load(f)
    print(len(lvis_val))
    print(lvis_val.keys())
    print(lvis_val['categories'][0])
    id2cat = {c["id"]: c for c in lvis_val["categories"]}
    all_cats = sorted(list(id2cat.keys()))
    label_set = torch.as_tensor(list(all_cats))
    splits = torch.split(label_set, 32)
    print(splits[0])

    for split in tqdm(splits):
        captions = [f"{clean_name(id2cat[l]['name'])}" for l in split.tolist()]
        tokenized = model.transformer.tokenizer.batch_encode_plus(captions, padding="longest", return_tensors="pt").to(
            device
        )
        encoded_text = model.transformer.text_encoder(**tokenized)

        # Transpose memory because pytorch's attention expects sequence first
        text_memory = encoded_text.last_hidden_state.transpose(0, 1)
        # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
        text_attention_mask = tokenized.attention_mask.ne(1).bool()

        # Resize the encoder hidden states to be of the same d_model as the decoder
        text_memory_resized = model.transformer.resizer(text_memory)

        text_memories.append((text_attention_mask, text_memory_resized, tokenized))




if __name__=='__main__':
    # path_to_json_gqa = '/mnt/data-nas/peizhi/data/mdetr_annotations'
    # observe the caption of LVIS
    # observe()

