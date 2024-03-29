# [NeurIPS 2022 Spotlight] RLIP: Relational Language-Image Pre-training for Human-Object Interaction Detection

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2209.01814)
[![GitHub Stars](https://img.shields.io/github/stars/JacobYuan7/RLIP?style=social)](https://github.com/JacobYuan7/RLIP)
[![GitHub Forks](https://img.shields.io/github/forks/JacobYuan7/RLIP)](https://github.com/JacobYuan7/RLIP)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FJacobYuan7%2FRLIP&count_bg=%235FC1D7&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false)](https://hits.seeyoufarm.com)
![visitors](https://visitor-badge.glitch.me/badge?page_id=JacobYuan7/RLIP)

## Updates
- **News**💥! The follow-up work [**RLIPv2: Fast Scaling of Relational Language-Image Pre-training**](https://arxiv.org/abs/2308.09351) is accepted to **ICCV 2023**. Its code have been released in [RLIPv2 repo](https://github.com/JacobYuan7/RLIPv2). 
- **Update on Jan. 19th, 2023**: I am uploading the code. Note that I changed all the path to prevent from possible information leakage. In order to run the code, you will need to configure the paths to match your own system. To do this, search for the **"/PATH/TO" placeholder** in the code and replace it with the appropriate file path on your system. ⭐⭐⭐Consider starring the repo! ⭐⭐⭐
- **Update on Jan. 16th, 2023**: I have uploaded the annotations and checkpoints. 
- **Update on Dec. 12th, 2022**: The code is under pre-release review in Alibaba Group, which will be made public as soon as possible.  
- **News**💥! **RLIP: Relational Language-Image Pre-training** is accepted to **NeurIPS 2022** as a **Spotlight** presentation (Top 5%)! Hope you will enjoy reading it.

<!---
**Update on Jan. 28th, 2023**: I updated the link of the pre-trained model in **Inference on Custom Images**. This updated one is fine-tuned on HICO. The old one is the pre-trained parameters. Sorry about the mistake.

The code is still under pre-release review because we encountered a minor technical issue. It will be out this week. Sincere apologies! -->

## Todo List
Note that if you can not get access to the links provided below, try using another browser or contact me by e-mail. 
- [x] 🎉 Release annotations for VG pre-training, HICO-DET few-shot, zero-shot and relation label noise. 
- [x] 🎉 Release checkpoints for pre-training, few-shot, zero-shot and fine-tuning.  
- [x] 🎉 Release code for pre-training, fine-tuning and inference.  
- [x] 🎉 Include support for inference on custom images.
- [ ] 🕘 Include support for Scene Graph Generation.  (It has been supported in [RLIPv2](https://github.com/JacobYuan7/RLIPv2).)

## Model Outline

This repo contains the implementation of various methods to resolve HOI detection (not limited to RLIP), aiming to serve as a benchmark for HOI detection. Below methods are included in this repo:
 - [RLIP-ParSe](https://arxiv.org/abs/2209.01814) (model name in the repo: RLIP-ParSe);
 - [ParSe](https://arxiv.org/abs/2209.01814) (model name in the repo: ParSe);
 - [RLIP-ParSeD](https://arxiv.org/abs/2209.01814) (model name in the repo: RLIP-ParSeD);
 - [ParSeD](https://arxiv.org/abs/2209.01814) (model name in the repo: ParSeD);
 - [OCN](https://github.com/JacobYuan7/OCN-HOI-Benchmark) (model name in the repo: OCN), which is a prior work of RLIP;  
 - [QPIC](https://github.com/hitachi-rd-cv/qpic) (model name in the repo: DETRHOI);
 - [QAHOI](https://github.com/cjw2021/QAHOI) (model name in the repo: DDETRHOI);
 - [CDN](https://github.com/YueLiao/CDN) (model name in the repo: CDN);
 
 <!--- a newly-proposed Cross-Modal Relation Detection method for HOI detection. -->


## Citation ##
If you find our work inspiring or our code/annotations useful to your research, please cite:
```bib
@inproceedings{Yuan2022RLIP,
  title={RLIP: Relational Language-Image Pre-training for Human-Object Interaction Detection},
  author={Yuan, Hangjie and Jiang, Jianwen and Albanie, Samuel and Feng, Tao and Huang, Ziyuan and Ni, Dong and Tang, Mingqian},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}

@inproceedings{Yuan2023RLIPv2,
  title={RLIPv2: Fast Scaling of Relational Language-Image Pre-training},
  author={Yuan, Hangjie and Zhang, Shiwei and Wang, Xiang and Albanie, Samuel and Pan, Yining and Feng, Tao and Jiang, Jianwen and Ni, Dong and Zhang, Yingya and Zhao, Deli},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}

@inproceedings{Yuan2022OCN,
  title={Detecting Human-Object Interactions with Object-Guided Cross-Modal Calibrated Semantics},
  author={Hangjie Yuan and Mang Wang and Dong Ni and Liangpeng Xu},
  booktitle={AAAI},
  year={2022}
}
```

## Inference on Custom Images
To facilitate the use of custom images without annotations, I have implemented a version of code that supports this. In terms of the pre-trained model, I am using the best-performing [RLIP-ParSe](https://zjueducn-my.sharepoint.com/:u:/g/personal/hj_yuan_zju_edu_cn/EfsAWI6hauxPoQXxPU96FrEBQO4J0079JQ3R3n5PA58inA?e=tmTD3a). To begin, please place your images in the folder `custom_imgs`. Then, you could try running the code below:
```shell
cd /PATH/TO/RLIP
# RLIP-ParSe
bash scripts/Inference_on_custom_imgs.sh
```
After successfully running the code, the generated results will be available in the folder `custom_imgs/result`. I have tested the code on a single Tesla A100 with batch sizes of 1, 2, 3 and 4. Note that, by default, we saved all the detection results (64 pairs). In most cases, it is possible to set a threshold for the **verb_scores** (multiplication of relation scores and object scores) in the saved results, which will enable their use in your own work. You can do it by yourself and tune the threshold for your work. As a recommendation, **0.25** (0.5*0.5) might be a good start to try out.


## Annotation Preparation
| Dataset | Setting | Download |
| ---------- | :-----------:  | :-----------:  |
| VG | Pre-training | [Link](https://1drv.ms/u/s!Areeng9FzbjiyE4ZsPNNhoIVDZnl?e=wUfANf) |
| HICO-DET | Few-shot 1%, 10% | [Link](https://1drv.ms/f/s!Areeng9FzbjixEBRdVQ7sE8sIyiW?e=LtdYUB) |
| HICO-DET | Zero-shot (UC-NF, UC-RF)\* | [Link](https://1drv.ms/f/s!Areeng9FzbjixEK_qV1cAwrLMW3A?e=QmX1Za) |
| HICO-DET | Relation Label Noise (10%, 30%, 50%) | [Link](https://1drv.ms/f/s!Areeng9FzbjixEHejFx_DmfzkeaH?e=3pCOKh) |

Note: ① \* Zero-shot (NF) do not need any HICO-DET annotations for fine-tuning, so we only provide training annotations for the UC-NF and UC-RF setting.

## Pre-training Dataset (Visual Genome) preparation
Firstly, we could download VG dataset from the [official link](https://visualgenome.org/api/v0/api_home.html), inclduing images Part I and Part II. (**Note: If the official website is not working, you can use the link that I provide: [Images](https://1drv.ms/u/s!Areeng9FzbjiyHQu_6NbSmsQf81D?e=BgMdXE) and [Images2](https://1drv.ms/u/s!Areeng9FzbjiyGUtQ8gfA7VVMvVp?e=aRzQeR).**) The annotations after pre-processing could be downloaded from the link above, which is used for pre-training. Note that this is generated from `scene_graphs.json` file by several pre-processing steps to remove redundant triplets. Also, several settings mentioned below also need the annotations that we provide. VG dataset and its corresponding annotations should be organized as follows:
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
| MDETR-ParSe | Modulated Detection | GoldG+ | ResNet-101 | DETR | [Link](https://1drv.ms/u/s!Areeng9Fzbjix1qGLPIu2q8IY_dg?e=eWklT4) |
| ParSeD | Object Detection | VG |ResNet-50 | DDETR | [Link](https://1drv.ms/u/s!Areeng9FzbjiyAmamusfMK2jdfc6?e=tDsgHe) |
| ParSeD | Object Detection | COCO |ResNet-50 | DDETR | [Link](https://1drv.ms/u/s!Areeng9FzbjiyGNWjuXnVAXdNrWA?e=b0piCY) |
| ParSe | Object Detection | COCO |ResNet-50 | DETR | [Link (Query128)](https://1drv.ms/u/s!Areeng9Fzbjix1XVGJu7lJsJxskL?e=511bye) <br> [Link (Query200)](https://1drv.ms/u/s!Areeng9Fzbjix1bnEOGreIQddcVn?e=YtNPMn) |
| ParSe | Object Detection | COCO |ResNet-101 | DETR | [Link (Query128)](https://1drv.ms/u/s!Areeng9Fzbjix1O2RD4I69uwd_JO?e=UVdY9Z) |
| RLIP-ParSeD | RLIP | VG | ResNet-50 | DDETR | [Link](https://1drv.ms/u/s!Areeng9Fzbjix15oKkjrK-FZ7NkM?e=We3Erx) |
| RLIP-ParSeD | RLIP | [COCO](https://zjueducn-my.sharepoint.com/:u:/g/personal/hj_yuan_zju_edu_cn/ERl3-qV1GWFNp2lFvO3nSCYBGovlWdrljrkavNy4LOjb-Q?e=g21Ksv) + VG | ResNet-50 | DDETR | [Link](https://1drv.ms/u/s!Areeng9Fzbjix1xo1hKAaBEgXVZy?e=6a4j6A) |
| RLIP-ParSe | RLIP | [COCO](https://zjueducn-my.sharepoint.com/:u:/g/personal/hj_yuan_zju_edu_cn/EZp2Qg5FS8hPlEd6uDVnma0BFkiMI2N312DKG_8xFf7aDQ?e=A3EIzo) + VG | ResNet-50 | DETR | [Link](https://1drv.ms/u/s!Areeng9Fzbjix13Fmxp9bYh5MRWT?e=Fn5nnC) |

With respect to the first, third and fourth line of the pre-trained weights, they are produced from the original codebase. For further reference, you could visit [DDETR](https://github.com/fundamentalvision/Deformable-DETR) and [MDETR](https://github.com/ashkamath/mdetr). The weights provided above are transformed from original codebases. With respect to the last three models' weights, optionally, you can pre-train the model yourself by running the corresponding script:
```shell
cd /PATH/TO/RLIP
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
| ParSeD | RLIP | COCO | ResNet-50 | DDETR | 29.12 / 22.23 / 31.17 | [Link](https://1drv.ms/u/s!Areeng9Fzbjix0jDkWKhSi7aILXe?e=zwOhHv) |
| ParSe | RLIP | COCO | ResNet-50 | DETR | 31.79 / 26.36 / 33.41 | [Link](https://1drv.ms/u/s!Areeng9Fzbjix0uLs6EXdYmtE-MU?e=ySevem) |
| ParSe | RLIP | COCO | ResNet-101 | DETR | 32.76 / 28.59 / 34.01 | [Link](https://1drv.ms/u/s!Areeng9Fzbjix0-5nBNt_JQ9d_5k?e=41ik0J) |
| RLIP-ParSeD | RLIP | VG | ResNet-50 | DDETR | 29.21 / 24.45 / 30.63 | [Link](https://1drv.ms/u/s!Areeng9Fzbjix0fOULAHQToKVwVQ?e=gBm5bP) |
| RLIP-ParSeD | RLIP | COCO + VG | ResNet-50 | DDETR | 30.70 / 24.67 / 32.50  | [Link](https://1drv.ms/u/s!Areeng9Fzbjix1Q31RbKTqpQ8UBp?e=e1bJaI) |
| RLIP-ParSe | RLIP | COCO + VG | ResNet-50 | DETR | 32.84 / 26.85 / 34.63 | [Link](https://1drv.ms/u/s!Areeng9Fzbjix0bfJeUVjr75KUNJ?e=ZKospd) |


### 2. Few-shot (0, 1%, 10%)
The scripts are identical to those for fully fine-tuning. The major difference is that we need to add `--few_shot_transfer 10 \` for 10% data of few-shot transfer and  `--few_shot_transfer 1 \` for 1% data of few-shot transfer. Note that we only fine-tune for 10 epochs with the lr dropping at 7th epoch. Thus, you need to change `--lr_drop` and `--epochs` in the script accordingly.
```shell
cd /PATH/TO/RLIP
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
| RLIP-ParSeD | RLIP | COCO + VG | ResNet-50 | DDETR | 0 | 13.92 / 11.20 / 14.73 | [Link](https://1drv.ms/u/s!Areeng9Fzbjix1xo1hKAaBEgXVZy?e=6a4j6A)\* |
| RLIP-ParSeD | RLIP | COCO + VG | ResNet-50 | DDETR | 1% | 18.30 / 16.22 / 18.92 | [Link](https://1drv.ms/f/s!Areeng9Fzbjiwzfdv4UH6S7hSlS3?e=Sk7Abv) |
| RLIP-ParSeD | RLIP | COCO + VG | ResNet-50 | DDETR | 10% | 22.09 / 15.89 / 23.94 | [Link](https://1drv.ms/f/s!Areeng9Fzbjiwzfdv4UH6S7hSlS3?e=Sk7Abv) |
| RLIP-ParSe | RLIP | COCO + VG | ResNet-50 | DETR | 0 | 15.40 / 15.08 / 15.50 | [Link](https://1drv.ms/u/s!Areeng9Fzbjix13Fmxp9bYh5MRWT?e=Fn5nnC)\* |
| RLIP-ParSe | RLIP | COCO + VG | ResNet-50 | DETR | 1% | 18.46 / 17.47 / 18.76 | [Link](https://1drv.ms/f/s!Areeng9Fzbjiwzfdv4UH6S7hSlS3?e=Sk7Abv) |
| RLIP-ParSe | RLIP | COCO + VG | ResNet-50 | DETR | 10% | 22.59 / 20.16 / 23.32   | [Link](https://1drv.ms/f/s!Areeng9Fzbjiwzfdv4UH6S7hSlS3?e=Sk7Abv) |

Note: ① \* means that the checkpoints are the same as the ones in the RLIP Pre-training table, since they do not involve any fine-tuning.

### 3. Zero-shot (NF, UC-RF, UC-NF)
With respect to NF setting, it is actually a testing procedure after loading the pre-trained weights. We could run the script below.
```shell
cd /PATH/TO/RLIP
# Zero-shot NF setting with RLIP-ParSe/RLIP-ParSeD
bash scripts/NF_Zero_shot.sh
```
With respect to UC-RF and UC-NF setting, training is required. We could run the script below by adding `--zero_shot_setting UC-RF \` or `--zero_shot_setting UC-NF \`. Note that for UC-NF setting, we only fine-tunes for 40 epochs (lr dropping at 30th epoch) to avoid overfitting. Thus, you need to change `--lr_drop 30 \` and `--epochs 40 \` in the script accordingly.
```shell
cd /PATH/TO/RLIP
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
| RLIP-ParSe | RLIP | COCO + VG | ResNet-50 | DETR | UC-RF | 30.52 / 19.19 / 33.35 | [Link](https://1drv.ms/f/s!Areeng9FzbjiwzoTAb42YUehBzUL?e=3JHpqP) |
| RLIP-ParSe | RLIP | COCO + VG | ResNet-50 | DETR | UC-NF | 26.19 / 20.27 / 27.67 | [Link](https://1drv.ms/f/s!Areeng9FzbjiwzoTAb42YUehBzUL?e=3JHpqP) |

## Evaluation
The mAP on HICO-DET under the Full set, Rare set and Non-Rare Set will be reported during the training process.

The results for the official evaluation of V-COCO must be obtained by the generated pickle file of detection results.
```shell
cd /PATH/TO/RLIP
python generate_vcoco_official.py \
        --param_path /PATH/TO/CHECKPOINT \
        --save_path vcoco.pickle \
        --hoi_path /PATH/TO/VCOCO/DATA \
```
Then you should run following codes after modifying the path to get the final performance:
```shell
cd /PATH/TO/RLIP
python datasets/vsrl_eval.py
```

## Acknowledgement
Part of this work's implemention refers to several prior works including [OCN](https://github.com/JacobYuan7/OCN-HOI-Benchmark), [QPIC](https://github.com/hitachi-rd-cv/qpic), [CDN](https://github.com/YueLiao/CDN), [DETR](https://github.com/facebookresearch/detr), [DDETR](https://github.com/fundamentalvision/Deformable-DETR) and [MDETR](https://github.com/ashkamath/mdetr).
