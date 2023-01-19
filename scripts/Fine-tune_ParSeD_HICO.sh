python models/ops/setup.py build install;
pip install -I transformers==4.5.1 --no-cache-dir --force-reinstall
pip install -r requirements_ParSeDETRHOI.txt;
pip install pkgs/pycocotools-2.0.2.tar.gz;
pip install submitit==1.3.0;
pip install timm;
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
    --pretrained /PATH/TO/params/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint_numrefpoints4_SepDDETRHOIv3_converted.pth \
    --output_dir /PATH/TO/logs/SepDDETRHOIv3_COCO_pretrained_pnms_public \
    --dim_feedforward 1024 \
    --epochs 60 \
    --lr_drop 40 \
    --num_queries 200 \
    --enc_layers 6 \
    --dec_layers 3 \
    --hoi \
    --dataset_file hico \
    --hoi_path /PATH/TO/data/hico_20160224_det \
    --num_obj_classes 80 \
    --num_verb_classes 117 \
    --backbone resnet50 \
    --set_cost_bbox 2.5 \
    --set_cost_giou 1 \
    --bbox_loss_coef 2.5 \
    --giou_loss_coef 1 \
    --num_workers 4 \
    --batch_size 4 \
    --num_feature_levels 4 \
    --with_box_refine \
    --ParSeD \
    --use_nms_filter \
    --load_backbone supervised \
    --save_ckp \
    --giou_verb_label \
    # --few_shot_transfer 10 \
    # --relation_label_noise 50 \
    # --few_shot_transfer 100 \
