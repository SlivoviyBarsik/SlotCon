#!/bin/bash

set -e
set -x

inter=$1
min_scale=$2
game=$6

data_dir="../atari_ds/${game}_128"
output_dir="./output/crops_${game}_${inter}_${min_scale}_800ep"

echo ${output_dir}

source venv/bin/activate

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
    main_pretrain.py \
    --dataset atari \
    --data-dir ${data_dir} \
    --output-dir ${output_dir} \
    \
    --arch spr_cnn \
    --dim-hidden 1024 \
    --dim-out 128 \
    --num-prototypes 32 \
    --teacher-momentum 0.99 \
    --teacher-temp 0.07 \
    --group-loss-weight 0.5 \
    \
    --image-size 128 \
    --batch-size 512 \
    --optimizer lars \
    --base-lr 1.0 \
    --weight-decay 1e-5 \
    --warmup-epoch 5 \
    --epochs 800 \
    --fp16 \
    \
    --print-freq 10 \
    --save-freq 20 \
    --auto-resume \
    --num-workers 8\
    \
    --interest-crop ${inter}\
    --min-scale ${min_scale}
    
