
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=7 python evaluate_depth_v2.py --height 448 --width 512 --scheduler_step_size 14 --dataset kaist --split kaist --num_epochs 50 --max_depth 50 \
                --batch_size 4 --frame_ids 0 --use_stereo --model_name Kaist_GBNet_thermal --scale_depth --models_to_load encoder depth \
                --data_path MTN_data --log_dir ./tmp --using_attention --using_diff --model GBNet_v2 --thermal --distill  \
                --load_weights_folder {pretrained_weight_path}
