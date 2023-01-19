# generate_vcoco_official for ParSe (SepDETRHOIv2)
# python generate_vcoco_official.py \
#         --param_path /PATH/TO/logs/SepDETRHOIv2_VCOCO_DETR_finetuned_COCO_query128_pnms/checkpoint0089.pth \
#         --save_path /PATH/TO/jacob/VCOCO_pickle/SepDETRHOIv2_vcoco_pnms.pickle \
#         --hoi_path /PATH/TO/data/v-coco \
#         --batch_size 4 \
#         --ParSe \
#         --num_obj_classes 81 \
#         --num_verb_classes 29 \
#         --dec_layers 3 \
#         --enc_layers 6 \
#         --num_queries 64 \
#        # --use_nms_filter \

# generate_vcoco_official for ParSeD (SepDDETRHOIv3)
# python generate_vcoco_official.py \
#         --param_path /PATH/TO/logs/SepDDETRHOIv3_VCOCO_vg_v1_pretrained_GIoUVb_pnms/checkpoint0059.pth \
#         --save_path /PATH/TO/jacob/VCOCO_pickle/SepDDETRHOIv3_vcoco_vg_v1.pickle \
#         --hoi_path /PATH/TO/data/v-coco \
#         --batch_size 4 \
#         --ParSeD \
#         --num_obj_classes 81 \
#         --num_verb_classes 29 \
#         --dec_layers 3 \
#         --enc_layers 6 \
#         --num_queries 200 \
#         --num_feature_levels 4 \
#         --with_box_refine \
#         --dim_feedforward 1024 \
# #         --param_path /PATH/TO/logs/SepDDETRHOIv3_VCOCO_COCO_pretrained_GIoUVb_pnms/checkpoint0059.pth \
# #         --save_path /PATH/TO/jacob/VCOCO_pickle/SepDDETRHOIv3_vc

# # generate_vcoco_official for RLIP-ParSeD (ParSeDDETRHOI)
# python generate_vcoco_official.py \
#         --param_path /PATH/TO/logs/ParSeD_VCOCO_Freq500_GIoUVb_PseuEuc03_COCO_bs128_50e_60e/checkpoint0059.pth \
#         --save_path /PATH/TO/jacob/VCOCO_pickle/ParSeDDETRHOI_vcoco_COCO_Init.pickle \
#         --hoi_path /PATH/TO/data/v-coco \
#         --batch_size 4 \
#         --RLIP_ParSeD \
#         --num_obj_classes 81 \
#         --num_verb_clas
#         --dec_layers 3 \
#         --enc_layers 6 \
#         --num_queries 200 \
#         --num_feature_levels 4 \
#         --with_box_refine \
#         --use_no_obj_token \
#         --dim_feedforward 1024 \
#         # --param_path /PATH/TO/logs/ParSeD_VCOCO_CEFocal_BiasT_Freq500_GIoUVb_PseuEuc03_50e_60e/checkpoint0059.pth \
#         # --save_path /PATH/TO/jacob/VCOCO_pickle/ParSeDDETRHOI_vcoco.pickle \


# generate_vcoco_official for RLIP-ParSe (ParSeDETRHOI)
python generate_vcoco_official.py \
        --param_path /PATH/TO/logs/ParSe_VCOCO_MDETR_HICO_90e/checkpoint0089.pth \
        --save_path /PATH/TO/jacob/VCOCO_pickle/ParSeDETRHOI_MDETR_R101_vcoco_WordsV3.pickle \
        --hoi_path /PATH/TO/data/v-coco \
        --batch_size 4 \
        --RLIP_ParSe \
        --num_obj_classes 81 \
        --num_verb_classes 29 \
        --dec_layers 3 \
        --enc_layers 6 \
        --num_queries 100 \
        --use_no_obj_token \
        --backbone resnet101 \
        # --use_nms_filter \

        # --param_path /PATH/TO/logs/ParSe_VCOCO_Que100_GIoUVb_PseuEuc03_aftNorm_COCO_bs128_150e_90e/checkpoint0089.pth \
        
        # --param_path /PATH/TO/logs/ParSe_VCOCO_Que100_WordsV3_COCO_bs128_150e_90e/checkpoint0089.pth \ Best
        # --save_path /PATH/TO/jacob/VCOCO_pickle/ParSeDETRHOI_vcoco_WordsV3.pickle \