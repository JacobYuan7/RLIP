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
# Pay attention to the learning rate if channging #nodes
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$WORLD_SIZE --node_rank=$RANK \
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py \
        --pretrained /PATH/TO/params/detr-r50-pre-hico_SepTransformerv2_query200_for_RLIP-ParSe.pth \
        --output_dir /PATH/TO/logs/ParSe_VG_BiasT_Freq500_GIoUVb_PseuEuc03_aftNorm_COCO_bs128_150e \
        --dataset_file vg \
        --vg_path /PATH/TO/data/VG \
        --backbone resnet50 \
        --set_cost_bbox 2.5 \
        --set_cost_giou 1 \
        --bbox_loss_coef 2.5 \
        --giou_loss_coef 1 \
        --batch_size 4 \
        --num_workers 4 \
        --lr_drop 100 \
        --epochs 150 \
        --load_backbone supervised \
        --dec_layers 3 \
        --enc_layers 6 \
        --lr 2e-4 \
        --lr_backbone 2e-5 \
        --text_encoder_lr 2e-5 \
        --num_queries 100 \
        --use_nms_filter \
        --RLIP_ParSe \
        --cross_modal_pretrain \
        --contrastive_loss_hdim 64 \
        --save_ckp \
        --schedule step \
        --subject_class \
        --use_no_obj_token \
        --obj_loss_type cross_entropy \
        --verb_loss_type focal \
        --negative_text_sampling 500 \
        --sampling_stategy freq \
        --giou_verb_label \
        --pseudo_verb \