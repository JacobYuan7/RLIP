python models/ops/setup.py build install;
pip install -I transformers==4.5.1 --no-cache-dir --force-reinstall
pip install -r requirements_ParSeDETRHOI.txt;
pip install pkgs/pycocotools-2.0.2.tar.gz;
pip install submitit==1.3.0;
pip install timm;
# export NCCL_DEBUG=INFO;
# export NCCL_IB_HCA=mlx5;
# export NCCL_IB_TC=136;
# export NCCL_IB_SL=5;
# export NCCL_IB_GID_INDEX=3;
# export TORCH_DISTRIBUTED_DETAIL=DEBUG;
# Pay attention to the learning rate if channging #nodes
# python -m torch.distributed.launch --nproc_per_node=8 --nnodes=$WORLD_SIZE --node_rank=$RANK \
#         --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --use_env main.py \
python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py \
        --pretrained /PATH/TO/params/detr-r50-pre-vcoco_SepTransformerv2_query200.pth \
        --output_dir /PATH/TO/logs/SepDETRHOIv2_VCOCO_DETR_finetuned_COCO_query200_Nopnms \
        --hoi \
        --dataset_file vcoco \
        --hoi_path /PATH/TO/data/v-coco \
        --num_obj_classes 81 \
        --num_verb_classes 29 \
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
        --lr 1e-4 \
        --lr_backbone 1e-5 \
        --dec_layers 3 \
        --ParSe \
        --save_ckp \
        --num_queries 100 \
        # --num_queries 64 \
        # --use_nms_filter \
        # --giou_verb_label \




