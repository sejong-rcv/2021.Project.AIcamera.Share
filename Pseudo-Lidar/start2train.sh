# --using_attention --using_diff --discriminator --distill --thermal  --compute --transloss 
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=3 python train.py --height 192 --width 640 --scheduler_step_size 14 \
#                 --batch_size 4 --frame_ids 0 --use_stereo --model_name stereo_woAttention_wodiff --png --data_path ./kitti_data --log_dir ./tmp --debug 

# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python train.py --height 448 --width 512 --scheduler_step_size 14 --dataset kaist --split kaist --num_epochs 50 --max_depth 50 \
#                 --batch_size 4 --frame_ids 0 --use_stereo --model_name Kaist_diff_distill_patch_pho--scale_depth\
#                 --data_path MTN_data --log_dir ./tmp --using_attention --using_diff --model DIFF --distill --compute --patchvlad --thermal \
#                 --load_weights_folder diffnet_640x192_ms #--debug #--num_workers 0

OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4 python train.py --height 448 --width 512 --scheduler_step_size 14 --dataset kaist --split kaist --num_epochs 50 --max_depth 50 \
                --batch_size 4 --frame_ids 0 --use_stereo --model_name Kaist_GBNet_thermal --scale_depth --models_to_load encoder depth \
                --data_path MTN_data --using_attention --using_diff --log_dir ./tmp --model GBNet --debug\
                --load_weights_folder tmp/Kaist_GBNet_RGB_AdamW/models/weights_11 #--debug #--num_workers 0
# OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 python train.py --height 448 --width 512 --scheduler_step_size 14 --dataset kaist --split kaist --num_epochs 50 --max_depth 50 \
#                 --batch_size 3 --frame_ids 0 --use_stereo --model_name Kaist_diff_RGB_GBNet_thermal_pre --thermal \
#                 --data_path MTN_data --log_dir ./tmp --model GBNet --load_weights_folder monodepth_pretrain #--debug