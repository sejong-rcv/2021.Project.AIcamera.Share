################################ Testing ################################
# DATA_PATH="../dataset/Luckydata"
DATA_PATH="../dataset/MTN"
# DATA_PATH="../dataset/potenit_real"
MODEL_NAME=MTN_hrt-T_256x256_bs32_T2RGB_v2.0.0
MODEL=hrt/hrt_tiny
CUDA_VISIBLE_DEVICES=5 python test.py --name $MODEL_NAME \
               --dataroot $DATA_PATH \
               --gpu_ids 0 \
               --load_size 256 \
               --nThreads 8\
               --output_nc 3\
               --no_flip \
               --serial_batches \
               --resize_or_crop resize \
               --cfg configs/$MODEL.yaml \
               --which_epoch 150\
            #    --save_result \
               

