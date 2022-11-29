# [NeurIPS 2022 Spotlight] RLIP: Relational Language-Image Pre-training for Human-Object Interaction Detection

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2209.01814)
[![GitHub Stars](https://img.shields.io/github/stars/JacobYuan7/RLIP?style=social)](https://github.com/JacobYuan7/RLIP)
[![GitHub Forks](https://img.shields.io/github/forks/JacobYuan7/RLIP)](https://github.com/JacobYuan7/RLIP)
![visitors](https://visitor-badge.glitch.me/badge?page_id=JacobYuan7/RLIP)

💥**News**! **RLIP: Relational Language-Image Pre-training** is accepted to ***NeurIPS 2022*** as a **spotlight** presentation (Top 5%)! Hope you will enjoy reading it.

This repo contains the implementation of various methods to resolve HOI detection (not limited to RLIP), aiming to serve as a benchmark for HOI detection. Below methods are included in this repo:
 - [RLIP-ParSe](https://arxiv.org/abs/2209.01814), a newly-proposed Cross-Modal Relation Detection method for HOI detection.
 - [ParSe](https://arxiv.org/abs/2209.01814)
 - [RLIP-ParSeD](https://arxiv.org/abs/2209.01814)
 - [ParSeD](https://arxiv.org/abs/2209.01814)
 - [OCN](https://github.com/JacobYuan7/OCN-HOI-Benchmark), which is a prior work of RLIP published by us;  
 - [QPIC](https://github.com/hitachi-rd-cv/qpic);  
 - [QAHOI](https://github.com/cjw2021/QAHOI);  
 - [CDN](https://github.com/YueLiao/CDN);


## Citation ##
If you find our work inspiring or our code useful to your research, please cite:
```bib
@inproceedings{Yuan2022RLIP,
  title={RLIP: Relational Language-Image Pre-training for Human-Object Interaction Detection},
  author={Yuan, Hangjie and Jiang, Jianwen and Albanie, Samuel and Feng, Tao and Huang, Ziyuan and Ni, Dong and Tang, Mingqian},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}

@inproceedings{Yuan2022OCN,
  title={Detecting Human-Object Interactions with Object-Guided Cross-Modal Calibrated Semantics},
  author={Hangjie Yuan and Mang Wang and Dong Ni and Liangpeng Xu},
  booktitle={AAAI},
  year={2022}
}
```

## Pre-training Dataset (Visual Genome) preparation
Firstly, we could download VG dataset from the [official link](https://visualgenome.org/api/v0/api_home.html), inclduing images Part I and Part II. The annotations after pre-processing could be downloaded from [this link](******Link to be added******). Note that this is generated from `scene_graphs.json` file by several pre-processing steps to remove redundant triplets. Also, several settings mentioned below also need the annotations that we provide. VG dataset and its corresponding annotations should be organized as follows:
```
VG
 |─ annotations
 |   |— scene_graphs_after_preprocessing.json
 |   :
 |— images
 |   |— 2409818.jpg
 |   |— n102412.jpg
 :   :
```

## Downstream Dataset preparation
### 1. HICO-DET
HICO-DET dataset can be downloaded [here](https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk). After finishing downloading, unpack the tarball (`hico_20160224_det.tar.gz`) to the `data` directory.

Instead of using the original annotations files, we use the annotation files provided by the PPDM authors. The annotation files can be downloaded from [here](https://drive.google.com/open?id=1WI-gsNLS-t0Kh8TVki1wXqc3y2Ow1f2R). The downloaded annotation files have to be placed as follows.
```
qpic
 |─ data
 │   └─ hico_20160224_det
 |       |─ annotations
 |       |   |─ trainval_hico.json
 |       |   |─ test_hico.json
 |       |   └─ corre_hico.npy
 :       :
```

### 2. V-COCO
First clone the repository of V-COCO from [here](https://github.com/s-gupta/v-coco), and then follow the instruction to generate the file `instances_vcoco_all_2014.json`. Next, download the prior file `prior.pickle` from [here](https://drive.google.com/drive/folders/10uuzvMUCVVv95-xAZg5KS94QXm7QXZW4). Place the files and make directories as follows.
```
qpic
 |─ data
 │   └─ v-coco
 |       |─ data
 |       |   |─ instances_vcoco_all_2014.json
 |       |   :
 |       |─ prior.pickle
 |       |─ images
 |       |   |─ train2014
 |       |   |   |─ COCO_train2014_000000000009.jpg
 |       |   |   :
 |       |   └─ val2014
 |       |       |─ COCO_val2014_000000000042.jpg
 |       |       :
 |       |─ annotations
 :       :
```
The annotation file has to be converted to the HOIA format. The conversion can be conducted as follows.
```
PYTHONPATH=data/v-coco \
        python convert_vcoco_annotations.py \
        --load_path data/v-coco/data \
        --prior_path data/v-coco/prior.pickle \
        --save_path data/v-coco/annotations
```
Note that only Python2 can be used for this conversion because `vsrl_utils.py` in the v-coco repository shows a error with Python3.

V-COCO annotations with the HOIA format, `corre_vcoco.npy`, `test_vcoco.json`, and `trainval_vcoco.json` will be generated to `annotations` directory.


Code will be available upon publication. So stay tuned!


