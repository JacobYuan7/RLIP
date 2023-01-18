python convert_parameters/convert_parameters_DDETR.py \
	--load_path /PATH/TO/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth \
	--save_path /PATH/TO/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint_numrefpoints4_SepDDETRHOIv3_converted.pth \         
	--ParSeD \
    --with_box_refine \
    --num_ref_points 2 \
