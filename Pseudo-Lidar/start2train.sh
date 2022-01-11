
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 python train.py --height 448 --width 512 --scheduler_step_size 14 --dataset kaist --split kaist --num_epochs 50 --max_depth 15 \
                --batch_size 4 --frame_ids 0 --use_stereo --model_name Kaist_GBNet_distill_v2 --scale_depth --models_to_load encoder \
                --data_path MTN_data --using_attention --using_diff --log_dir ./tmp --model GBNet_v2 --self_guided --compute --thermal\
                --load_weights_folder {pretrained_weight_path}
