#!/bin/bash
file1="maple"
stdfile='docq_yellow_std'
date='07032022'
for var in 2400
do
    python trainStructureTransfer.py \
    --style_name ../data/style/${file1}.png \
    --std_name ../data/style/${stdfile}.png \
    --batchsize 6 --Straining_num ${var} \
    --step1_epochs 80 --step2_epochs 40 --step3_epochs 80 \
    --scale_num 2 \
    --Sanglejitter \
    --save_path ../save --save_name ${file1}_${var}_${date} \
    --text_path ../data/rawtext/yaheiB/train --text_datasize 708 \
    --load_GB_name ../save/GB-iccv.ckpt \
    --gpu

done
