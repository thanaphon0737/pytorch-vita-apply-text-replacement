#!/bin/bash
pic="docq_yellow_std"
file1="docq_yellow"
stdfile='docq_yellow_std'
date='05042022-250im-10001ep-rec-200-sadv-0.1'
for var in 6
do
    python trainStructureTransfer_edit.py \
    --text_name ../data/rawtext/yaheiB/val/test1a_data/${pic}.png \
    --name ${file1} \
    --style_name ../data/style/${file1}.png \
    --std_name ../data/style/${stdfile}.png \
    --batchsize 6 --Straining_num ${var} \
    --step1_epochs 10001 --step2_epochs 40 --step3_epochs 80 \
    --scale_num 2 \
    --Sanglejitter \
    --save_path ../save --save_name ${file1}_${var}_${date} \
    --text_path ../data/rawtext/yaheiB/train --text_datasize 708 \
    --load_GB_name ../save/GB-iccv.ckpt \
    --gpu

done
