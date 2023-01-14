# [NeurIPS 2022 Spotlight] RLIP: Relational Language-Image Pre-training for Human-Object Interaction Detection

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2209.01814)
[![GitHub Stars](https://img.shields.io/github/stars/JacobYuan7/RLIP?style=social)](https://github.com/JacobYuan7/RLIP)
[![GitHub Forks](https://img.shields.io/github/forks/JacobYuan7/RLIP)](https://github.com/JacobYuan7/RLIP)
![visitors](https://visitor-badge.glitch.me/badge?page_id=JacobYuan7/RLIP)

## Todo List and Updates
- [x] 🎉 Annotations for VG pre-training, HICO-DET few-shot, zero-shot and relation label noise. 
- [x] 🎉 Checkpoints for pre-training, few-shot, zero-shot and fine-tuning.  
- [ ] 🕘 Code for pre-training, fine-tuning and inference.  
- [ ] 🕘 Support for Scene Graph Generation.  


- **Update on Jan. 13th, 2023**: I am uploading the annotations and checkpoints. The code is still under pre-release review because we encountered a technical issue. It should be out in 1-2 weeks. Sincere apologies!
- **Update on Dec. 12th, 2022**: The code is under reviewing in Alibaba Group, which will be made public as soon as possible.
- 💥**News**! **RLIP: Relational Language-Image Pre-training** is accepted to ***NeurIPS 2022*** as a **spotlight** presentation (Top 5%)! Hope you will enjoy reading it.

## Outline

This repo contains the implementation of various methods to resolve HOI detection (not limited to RLIP), aiming to serve as a benchmark for HOI detection. Below methods are included in this repo:
 - [RLIP-ParSe](https://arxiv.org/abs/2209.01814), a newly-proposed Cross-Modal Relation Detection method for HOI detection.
 - [ParSe](https://arxiv.org/abs/2209.01814)
 - [RLIP-ParSeD](https://arxiv.org/abs/2209.01814)
 - [ParSeD](https://arxiv.org/abs/2209.01814)
 - [OCN](https://github.com/JacobYuan7/OCN-HOI-Benchmark), which is a prior work of RLIP finished by us;  
 - [QPIC](https://github.com/hitachi-rd-cv/qpic);  
 - [QAHOI](https://github.com/cjw2021/QAHOI);  
 - [CDN](https://github.com/YueLiao/CDN);


## Citation ##
If you find our work inspiring or our code/annotations useful to your research, please cite:
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

## Annotation Preparation
| Dataset | Setting | Download |
| ---------- | :-----------:  | :-----------:  |
| VG | Pre-training | [Link](https://zjueducn-my.sharepoint.com/:u:/g/personal/hj_yuan_zju_edu_cn/EWEPvw_EEttNt4TNHABDWbgB0S4LBPzlxvPidh_MhEEUTQ?e=j9gBjk) |
| HICO-DET | Few-shot 1%, 10% | [Link](https://zjueducn-my.sharepoint.com/:f:/g/personal/hj_yuan_zju_edu_cn/Eh7UufFbB_5Dutvr66g-t6sBn5wCeA0uzMwiy8mUxaD50g?e=IKB3SD) |
| HICO-DET | Zero-shot (UC-NF, UC-RF)\* | [Link](https://zjueducn-my.sharepoint.com/:f:/g/personal/hj_yuan_zju_edu_cn/Ev9BzZxOlT5Mt04wOpIHA5kBP2eA6fijjweI_kh9WN3MUw?e=jMJmu6) |
| HICO-DET | Relation Label Noise (10%, 30%, 50%) | [Link](https://zjueducn-my.sharepoint.com/:f:/g/personal/hj_yuan_zju_edu_cn/EnG5dDpkompHvy5ez82NQN4Bu9XDpvl9d9P73W7fQEmn7w?e=4dqaZ2) |

Note: ① \* Zero-shot (NF) do not need any HICO-DET annotations for fine-tuning, so we only provide training annotations for the UC-NF and UC-RF setting.

## Pre-training Dataset (Visual Genome) preparation
Firstly, we could download VG dataset from the [official link](https://visualgenome.org/api/v0/api_home.html), inclduing images Part I and Part II. The annotations after pre-processing could be downloaded from the link above, which is used for pre-training. Note that this is generated from `scene_graphs.json` file by several pre-processing steps to remove redundant triplets. Also, several settings mentioned below also need the annotations that we provide. VG dataset and its corresponding annotations should be organized as follows:
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

## RLIP Pre-training
Since RLIP pre-trained on VG and COCO dataset, we provide a series of pre-trained weights for you to use. Weights in the table below are used to initialize ParSe/ParSeD/RLIP-ParSe/RLIP-ParSeD for pre-training or fine-tuning.
| Model | Pre-training Paradigm | Pre-training Dataset | Backbone | Base Detector | Download |
| ---------- | :-----------:  | :-----------:  | :-----------: | :-----------: | :-----------: |
| MDETR-ParSe | Modulated Detection | GoldG+ | ResNet-101 | DETR | [Link](https://zjueducn-my.sharepoint.com/:u:/g/personal/hj_yuan_zju_edu_cn/EU6opo5zQR5Erculzyqg8l0BuGfcYfRp-9pyhu30SqB9qQ?e=q5WX1d) |
| ParSeD | Object Detection | VG |ResNet-50 | DDETR | [Link](https://zjueducn-my.sharepoint.com/:u:/g/personal/hj_yuan_zju_edu_cn/EWl5gPmxYbFKkus7wt5J2TYBveMO0MswfcojqlemPcX8ww?e=XbkBcv) |
| ParSeD | Object Detection | COCO |ResNet-50 | DDETR | [Link](******Link to be added******) |
| ParSe | Object Detection | COCO |ResNet-50 | DETR | [Link (Query128)](https://zjueducn-my.sharepoint.com/:u:/g/personal/hj_yuan_zju_edu_cn/EUTlGpsdSf1MsIHaEsQ7V7oBX_ufXXSMNm657jEeUdTQaQ?e=1g63Xe) <br> [link (Query200)](https://zjueducn-my.sharepoint.com/:u:/g/personal/hj_yuan_zju_edu_cn/EbA-wS49GOpOvxQK3HmIi_UBCuRiln15WDRaCO18xFq5KA?e=INg3Us) |
| ParSe | Object Detection | COCO |ResNet-101 | DETR | [Link (Query128)](https://zjueducn-my.sharepoint.com/:u:/g/personal/hj_yuan_zju_edu_cn/ESEK1Jv5tnhBrvDcGqtYuxwBjZFx2ws-K-qeo9J-LsmdRw?e=d6prKQ) |
| RLIP-ParSeD | RLIP | VG | ResNet-50 | DDETR | [Link](https://zjueducn-my.sharepoint.com/:u:/g/personal/hj_yuan_zju_edu_cn/EX2W9LDC7aJHqFCCEvlBz5gBV_kw3nokkKO6eZA8468BlQ?e=7GByjS) |
| RLIP-ParSeD | RLIP | [COCO](*) + VG | ResNet-50 | DDETR | [Link](https://zjueducn-my.sharepoint.com/:u:/g/personal/hj_yuan_zju_edu_cn/ESN0gmIqMdtNlfvukKrxOJkBTF_Oz2R32imDfkShe8ZRAA?e=gUQ8pe) |
| RLIP-ParSe | RLIP | [COCO](*) + VG | ResNet-50 | DETR | [Link](https://zjueducn-my.sharepoint.com/:u:/g/personal/hj_yuan_zju_edu_cn/EfsAWI6hauxPoQXxPU96FrEBQO4J0079JQ3R3n5PA58inA?e=tmTD3a) |

With respect to the first, third and fourth line of the pre-trained weights, they are produced from the original codebase. For further reference, you could visit [DDETR](https://github.com/fundamentalvision/Deformable-DETR) and [MDETR](https://github.com/ashkamath/mdetr). The weights provided above are transformed from original codebases. With respect to the last three models' weights, optionally, you can pre-train the model yourself by running the corresponding script:
```shell
# RLIP-ParSe
bash scripts/Pre-train_RLIP-ParSe_VG.sh
# RLIP-ParSeD
bash scripts/Pre-train_RLIP-ParSeD_VG.sh
```
Note that above scripts contain the installation of dependencies, which could be done independently. For the `--pretrained` parameter in script, you could ignore it to pre-train from scratch or use ParSeD parameters pre-trained on COCO.

### 1. Fully-finetuning

Weights in the table below are fully-fined weights of ParSe/ParSeD/RLIP-ParSe/RLIP-ParSeD using pre-trained weights from the above table.
| Model | Pre-training Paradigm | Pre-training Dataset | Backbone | Base Detector | Full / Rare / Non-Rare | Download |
| ---------- | :-----------:  | :-----------:  | :-----------: | :-----------: | :-----------: | :-----------: |
| ParSeD | RLIP | COCO | ResNet-50 | DDETR | 29.12 / 22.23 / 31.17 | [Link](******Link to be added******) |
| ParSe | RLIP | COCO | ResNet-50 | DETR | 31.79 / 26.36 / 33.41 | [Link](******Link to be added******) |
| ParSe | RLIP | COCO | ResNet-101 | DETR | 32.76 / 28.59 / 34.01 | [Link](******Link to be added******) |
| RLIP-ParSeD | RLIP | VG | ResNet-50 | DDETR | 29.21 / 24.45 / 30.63 | [Link](******Link to be added******) |
| RLIP-ParSeD | RLIP | COCO + VG | ResNet-50 | DDETR | 30.70 / 24.67 / 32.50  | [Link](******Link to be added******) |
| RLIP-ParSe | RLIP | COCO + VG | ResNet-50 | DETR | 32.84 / 26.85 / 34.63 | [Link](******Link to be added******) |


### 2. Few-shot (0, 1%, 10%)
The scripts are identical to those for fully fine-tuning. The major difference is that we need to add `--few_shot_transfer 10 \` for 10% data of few-shot transfer and  `--few_shot_transfer 1 \` for 1% data of few-shot transfer. Note that we only fine-tune for 10 epochs with the lr dropping at 7th epoch. Thus, you need to change `--lr_drop` and `--epochs` in the script accordingly.
```shell
# RLIP-ParSeD on HICO
bash scripts/Fine-tune_RLIP-ParSeD_HICO.sh
# RLIP-ParSe on HICO
bash scripts/Fine-tune_RLIP-ParSe_HICO.sh
# ParSe on HICO
bash scripts/Fine-tune_ParSe_HICO.sh
# ParSeD on HICO
bash scripts/Fine-tune_ParSeD_HICO.sh
```
When there is no extra data provided (0 percent of few-shot transfer), please refer to zero-shot NF setting, but performance is present here.
| Model | Pre-training Paradigm | Pre-training Dataset | Backbone | Base Detector | Data | Full / Rare / Non-Rare | Download |
| ---------- | :-----------:  | :-----------:  | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| RLIP-ParSeD | RLIP | COCO + VG | ResNet-50 | DDETR | 0 | 13.92 / 11.20 / 14.73 | [Link](https://zjueducn-my.sharepoint.com/:u:/g/personal/hj_yuan_zju_edu_cn/ESN0gmIqMdtNlfvukKrxOJkBTF_Oz2R32imDfkShe8ZRAA?e=gUQ8pe)\* |
| RLIP-ParSeD | RLIP | COCO + VG | ResNet-50 | DDETR | 1% | 18.30 / 16.22 / 18.92 | [Link](******Link to be added******) |
| RLIP-ParSeD | RLIP | COCO + VG | ResNet-50 | DDETR | 10% | 22.09 / 15.89 / 23.94 | [Link](******Link to be added******) |
| RLIP-ParSe | RLIP | COCO + VG | ResNet-50 | DETR | 0 | 15.40 / 15.08 / 15.50 | [Link](https://zjueducn-my.sharepoint.com/:u:/g/personal/hj_yuan_zju_edu_cn/EfsAWI6hauxPoQXxPU96FrEBQO4J0079JQ3R3n5PA58inA?e=tmTD3a)\* |
| RLIP-ParSe | RLIP | COCO + VG | ResNet-50 | DETR | 1% | 18.46 / 17.47 / 18.76 | [Link](******Link to be added******) |
| RLIP-ParSe | RLIP | COCO + VG | ResNet-50 | DETR | 10% | 22.59 / 20.16 / 23.32   | [Link](******Link to be added******) |

Note: ① \* means that the checkpoints are the same as the ones in the RLIP Pre-training table, since they do not involve any fine-tuning.

### 3. Zero-shot (NF, UC-RF, UC-NF)
With respect to NF setting, it is actually a testing procedure after loading the pre-trained weights. We could run the script below.
```shell
# Zero-shot NF setting with RLIP-ParSe/RLIP-ParSeD
bash scripts/NF_Zero_shot.sh
```
With respect to UC-RF and UC-NF setting, training is required. We could run the script below by adding `--zero_shot_setting UC-RF \` or `--zero_shot_setting UC-NF \`. Note that for UC-NF setting, we only fine-tunes for 40 epochs (lr dropping at 30th epoch) to avoid overfitting. Thus, you need to change `--lr_drop 30 \` and `--epochs 40 \` in the script accordingly.
```shell
# RLIP-ParSeD on HICO
bash scripts/Fine-tune_RLIP-ParSeD_HICO.sh
# RLIP-ParSe on HICO
bash scripts/Fine-tune_RLIP-ParSe_HICO.sh
# ParSe on HICO
bash scripts/Fine-tune_ParSe_HICO.sh
# ParSeD on HICO
bash scripts/Fine-tune_ParSeD_HICO.sh
```
| Model | Pre-training Paradigm | Pre-training Dataset | Backbone | Base Detector | Setting | Full / Rare / Non-Rare | Download |
| ---------- | :-----------:  | :-----------:  | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| RLIP-ParSe | RLIP | COCO + VG | ResNet-50 | DETR | UC-RF | 30.52 / 19.19 / 33.35 | [Link](******Link to be added******) |
| RLIP-ParSe | RLIP | COCO + VG | ResNet-50 | DETR | UC-NF | 26.19 / 20.27 / 27.67 | [Link](******Link to be added******) |

## Evaluation
The mAP on HICO-DET under the Full set, Rare set and Non-Rare Set will be reported during the training process.

The results for the official evaluation of V-COCO must be obtained by the generated pickle file of detection results.
```shell
python generate_vcoco_official.py \
        --param_path /PATH/TO/CHECKPOINT \
        --save_path vcoco.pickle \
        --hoi_path /PATH/TO/VCOCO/DATA \
```
Then you should run following codes after modifying the path to get the final performance:
```shell
python datasets/vsrl_eval.py
```

## Acknowledgement
Part of this work's implemention refers to several prior works including [OCN](https://github.com/JacobYuan7/OCN-HOI-Benchmark), [QPIC](https://github.com/hitachi-rd-cv/qpic), [CDN](https://github.com/YueLiao/CDN), [DETR](https://github.com/facebookresearch/detr), [DDETR](https://github.com/fundamentalvision/Deformable-DETR) and [MDETR](https://github.com/ashkamath/mdetr).
