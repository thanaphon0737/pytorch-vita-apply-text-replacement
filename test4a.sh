#!/bin/bash
pic="docq_thin_sk"
file1="docq_thin_dis"
stdfile='docq_thin_sk'
sadv='0.1'
bsize='64'
epoch='100'
train_num='13312'
for rec in 100
do
    python trainStructureTransfer_edit.py \
    --text_name ../data/rawtext/yaheiB/val/test1a_data/${pic}.png \
    --name ${file1} \
    --style_name ../data/style/${file1}.png \
    --std_name ../data/style/${stdfile}.png \
    --batchsize ${bsize} --Straining_num ${train_num} --subimg_size 60\
    --step1_epochs ${epoch} --step2_epochs 40 --step3_epochs 80 \
    --l1 ${rec} --sadv ${sadv} \
    --scale_num 2 \
    --Sanglejitter \
    --save_path ../save --save_name newtest_${file1}_${bsize}-${epoch}ep-rec-${rec}-sadv-${sadv} \
    --text_path ../data/rawtext/yaheiB/train --text_datasize 708 \
    --load_GB_name ../save/GB-iccv.ckpt \
    --gpu

done
