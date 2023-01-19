python models/ops/setup.py build install;
pip install -I transformers==4.5.1 --no-cache-dir --force-reinstall
pip install -r requirements_ParSeDETRHOI.txt;
pip install pkgs/pycocotools-2.0.2.tar.gz;
pip install submitit==1.3.0;
pip install timm;
export NCCL_DEBUG=INFO;
export NCCL_IB_HCA=mlx5;
export NCCL_IB_TC=136;
export NCCL_IB_SL=5;
export NCCL_IB_GID_INDEX=3;
export TORCH_DISTRIBUTED_DETAIL=DEBUG;
# MDETR needs ResNet-101, plain RLIP uses ResNet-50
# python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$WORLD_SIZE --node_rank=$RANK \
#         --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py \
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
        --pretrained /PATH/TO/params/ParSe_VG_BiasT_Freq500_GIoUVb_PseuEuc03_aftNorm_COCO_bs128_150e_checkpoint0149.pth \
        --output_dir /PATH/TO/logs/ParSe_HICO_GIoUVb_PseuEuc03_aftNorm_COCO_bs128_150e_90e_Noi50 \
        --dataset_file hico \
        --hoi_path /PATH/TO/data/hico_20160224_det \
        --hoi \
        --backbone resnet50 \
        --set_cost_bbox 2.5 \
        --set_cost_giou 1 \
        --bbox_loss_coef 2.5 \
        --giou_loss_coef 1 \
        --batch_size 4 \
        --num_workers 4 \
        --lr_drop 60 \
        --epochs 90 \
        --load_backbone supervised \
        --dec_layers 3 \
        --enc_layers 6 \
        --lr 1e-4 \
        --lr_backbone 1e-5 \
        --text_encoder_lr 1e-5 \
        --use_nms_filter \
        --contrastive_loss_hdim 64 \
        --RLIP_ParSe \
        --obj_loss_type cross_entropy \
        --verb_loss_type focal \
        --schedule step \
        --save_ckp \
        --use_no_obj_token \
        --negative_text_sampling 0 \
        --sampling_stategy freq \
        --num_queries 64 \
        --giou_verb_label \
        --few_shot_transfer 100 \
        # --relation_label_noise 50 \
        # --zero_shot_setting UC-NF \