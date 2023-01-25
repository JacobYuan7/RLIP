# inference_on_custom_imgs_hico.py for RLIP-ParSe (ParSeDETRHOI)
python inference_on_custom_imgs_hico.py \
        --param_path /PATH/TO/ParSe_HICO_GIoUVb_PseuEuc03_aftNorm_COCO_bs128_150e_90e_checkpoint0087.pth \
        --save_path custom_imgs/result/custom_imgs.pickle \
        --batch_size 1 \
        --RLIP_ParSe \
        --num_obj_classes 80 \
        --num_verb_classes 117 \
        --dec_layers 3 \
        --enc_layers 6 \
        --num_queries 64 \
        --use_no_obj_token \
        --backbone resnet50 \