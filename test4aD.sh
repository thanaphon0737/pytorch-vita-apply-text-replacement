#!/bin/bash
pic="docq_thin_sk4_reduce"
file1="docq_thin_dis"
stdfile='docq_thin_sk4'
sadv='0.1'
bsize='6'
epoch='5000'
train_num='0'
subimage='120'
for rec in 100
do
    python trainStructureTransfer_saveD.py \
    --text_name ../data/rawtext/yaheiB/val/test1a_data/${pic}.png \
    --name ${file1} \
    --style_name ../data/style/${file1}.png \
    --std_name ../data/style/${stdfile}.png \
    --batchsize ${bsize} --Straining_num ${train_num} --subimg_size ${subimage}\
    --step1_epochs ${epoch}\
    --l1 ${rec} --sadv ${sadv} \
    --scale_num 2 \
    --Sanglejitter \
    --save_path ../save --save_name sk4_${file1}_${train_num}_${bsize}-${subimage}-${epoch}ep-rec-${rec}-sadv-${sadv} \
    --text_path ../data/rawtext/yaheiB/train --text_datasize 708 \
    --gpu

done
