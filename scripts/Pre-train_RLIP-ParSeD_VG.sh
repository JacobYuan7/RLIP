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
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=$WORLD_SIZE --node_rank=$RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py \
    --pretrained /PATH/TO/params/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint_numrefpoints4_SepDDETRHOIv3_converted.pth \
    --output_dir /PATH/TO/logs/ParSeD_CEFocal_BiasT_Freq500_PseuEuc03_COCO_bs128_50e \
    --dim_feedforward 1024 \
    --epochs 50 \
    --lr_drop 40 \
    --num_queries 200 \
    --enc_layers 6 \
    --dec_layers 3 \
    --dataset_file vg \
    --vg_path /PATH/TO/data/VG \
    --backbone resnet50 \
    --set_cost_bbox 2.5 \
    --set_cost_giou 1 \
    --bbox_loss_coef 2.5 \
    --giou_loss_coef 1 \
    --num_workers 4 \
    --batch_size 4 \
    --num_feature_levels 4 \
    --with_box_refine \
    --use_nms_filter \
    --load_backbone supervised \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --text_encoder_lr 2e-5 \
    --cross_modal_pretrain \
    --RLIP_ParSeD \
    --subject_class \
    --use_no_obj_token \
    --verb_loss_type focal \
    --save_ckp \
    --schedule step \
    --negative_text_sampling 500 \
    --obj_loss_type cross_entropy \
    --sampling_stategy freq \
    --giou_verb_label \
    --pseudo_verb \