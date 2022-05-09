#!/bin/bash
for var in anime-c stone-c yellow holl2
do
    python trainStructureTransfer.py \
    --style_name ../data/style/${var}.png \
    --batchsize 6 --Straining_num 700 \
    --step1_epochs 30 --step2_epochs 40 --step3_epochs 80 \
    --scale_num 4 \
    --Sanglejitter \
    --save_path ../save --save_name ${var} \
    --text_path ../data/rawtext/yaheiB/train --text_datasize 708 \
    --load_GB_name ../save/GB-iccv.ckpt \
    --gpu

    python trainTextureTransfer.py \
    --style_name ../data/style/${var}.png \
    --batchsize 2 --Ttraining_num 400 \
    --texture_step1_epochs 40 \
    --style_loss \
    --save_path ../save --save_name ${var} \
    --text_path ../data/rawtext/yaheiB/train --text_datasize 708 \
    --load_GS_name ../save/${var}-GS.ckpt \
    --gpu
done
