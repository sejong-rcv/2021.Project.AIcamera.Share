# DATA_PATH="../dataset/MTN"
DATA_PATH="../dataset/potenit"
# DATA_PATH="../dataset/Luckydata"
MODEL_NAME=Potenit_hrt-T_256x256_bs32_T2RGB_v2.0.0
PROJECTNAME="Potenit_Translation"
# MODEL_NAME=debug
MODEL=hrt/hrt_tiny
# torch.distributed.launch --nproc_per_node=2 
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=5 python train.py --gpu_ids 0 \
                --photometric \
                --name $MODEL_NAME \
                --project_name $PROJECTNAME \
                --batch_size 32 \
                --load_size 256 \
                --crop_size 256 \
                --dataroot $DATA_PATH \
                --resize_or_crop resize \
                --nThreads 16 \
                --display_freq 300 \
                --niter 100  \
                --niter_decay 50 \
                --lambda_feat 10 \
                --output_nc 3 \
                --cfg configs/$MODEL.yaml \
                --isdecoder \
                # --debug \
                
