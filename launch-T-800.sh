#!/bin/bash
for var in  yellow holl2
do
    python trainTextureTransfer.py \
    --style_name ../data/style/${var}.png \
    --batchsize 2 --Ttraining_num 800 \
    --texture_step1_epochs 40 \
    --style_loss \
    --save_path ../save --save_name ${var}800 \
    --text_path ../data/rawtext/yaheiB/train --text_datasize 708 \
    --load_GS_name ../save/blue2-GS.ckpt \
    --gpu
done
