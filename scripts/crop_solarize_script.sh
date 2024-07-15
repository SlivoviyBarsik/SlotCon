#!/bin/bash

set -e
set -x

glob_crop=$1
solarize_p=$2

data_dir="../atari_ds/assault_128"
output_dir="./output/assault_${glob_crop}_${solarize_p}_800ep"

echo ${output_dir}

source venv/bin/activate

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
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
    --num-workers 8 \
    --global-crop ${glob_crop} \
    --solarize-p ${solarize_p}



