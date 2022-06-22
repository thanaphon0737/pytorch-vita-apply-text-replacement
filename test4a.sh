#!/bin/bash
# pic='0_BroadwayRegular_35_1_0_BroadwayRegular_dis_sk4'
file1="0_BroadwayRegular_dis"
stdfile='0_BroadwayRegular_35_1_0_BroadwayRegular_dis_sk4'
sadv='0.1'
bsize='3'
epoch='3000'
train_num='35'
subimage='204'
rec='100'
for pic in 0_BroadwayRegular_35_1_0_BroadwayRegular_dis_sk4 0_BroadwayRegular_55_1_0_BroadwayRegular_dis_sk4 0_BroadwayRegular_73_1_0_BroadwayRegular_dis_sk4 0_BroadwayRegular_75_1_0_BroadwayRegular_dis_sk4 0_BroadwayRegular_77_1_0_BroadwayRegular_dis_sk4
do
    python trainStructureTransfer_split.py \
    --text_name ../data/rawtext/yaheiB/val/test1a_data/${pic}.png \
    --name ${file1} \
    --style_name ../data/style/${file1}.png \
    --std_name ../data/style/${pic}.png \
    --batchsize ${bsize} --Straining_num ${train_num} --subimg_size ${subimage}\
    --step1_epochs ${epoch}\
    --l1 ${rec} --sadv ${sadv} \
    --scale_num 2 \
    --Sanglejitter \
    --save_path ../save --save_name sk4_${pic}_${bsize}-${subimage}-${epoch}ep-rec-${rec}-sadv-${sadv} \
    --text_path ../data/rawtext/yaheiB/train --text_datasize 708 \
    --gpu 

    python verify.py \
    --file_name sk4_${pic}_${bsize}-${subimage}-${epoch}ep-rec-${rec}-sadv-${sadv} \
    --pic ${file1} 

    python clear_file.py \
    --file_name sk4_${pic}_${bsize}-${subimage}-${epoch}ep-rec-${rec}-sadv-${sadv} 
done
