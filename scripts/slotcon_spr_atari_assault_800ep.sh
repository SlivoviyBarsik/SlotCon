#!/bin/bash

set -e
set -x

arch=$1
dim_out=$2
dim_hidden=$3

data_dir="../atari_ds/assault_128"
output_dir="./output/slotcon_spr_atari_assault_${arch}_${dim_out}_${dim_hidden}_800ep"

echo ${output_dir}

source venv/bin/activate

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 \
    main_pretrain.py \
    --dataset atari \
    --data-dir ${data_dir} \
    --output-dir ${output_dir} \
    \
    --arch ${arch} \
    --dim-hidden ${dim_hidden} \
    --dim-out ${dim_out} \
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
    --save-freq 10 \
    --auto-resume \
    --num-workers 8\
    \
    --slotcon spr \
    --spr-skip 1 \
    --spr-lambda 0.33
