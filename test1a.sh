#!/bin/bash
for var in 0_f 1_f 2_f 3_f 4_f 5_f 6_f 7_f 8_f 9_f
do
    python trainStructureTransfer.py \
    --style_name ../data/style/${var}.png \
    --batchsize 6 --Straining_num 10 \
    --step1_epochs 30 --step2_epochs 40 --step3_epochs 80 \
    --scale_num 4 \
    --Sanglejitter \
    --save_path ../save --save_name ${var} \
    --text_path ../data/rawtext/yaheiB/train --text_datasize 708 \
    --load_GB_name ../save/GB-iccv.ckpt \
    --gpu

done
