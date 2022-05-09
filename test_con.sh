#!/bin/bash
file1="docq"
model_load="docq_1000"
var=500
varSum=1000+500
python trainStructureTransfer.py \
--style_name ../data/style/${file1}.png \
--batchsize 6 --Straining_num ${var} \
--step1_epochs 30 --step2_epochs 40 --step3_epochs 80 \
--scale_num 4 \
--Sanglejitter \
--save_path ../save --save_name ${file1}_${varSum} \
--text_path ../data/rawtext/yaheiB/train --text_datasize 708 \
--load_GB_name ../save/GB-iccv.ckpt \
--structure_load ../save/${model_load}-GS.ckpt
--gpu

