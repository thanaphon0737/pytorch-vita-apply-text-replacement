#!/bin/bash
file1="anime-c"
python trainStructureTransfer.py \
--style_name ../data/style/${file1}.png \
--batchsize 6 --Straining_num 1 \
--step1_epochs 30 --step2_epochs 40 --step3_epochs 80 \
--scale_num 4 \
--Sanglejitter \
--save_path ../save --save_name ${file1} \
--text_path ../data/rawtext/yaheiB/train --text_datasize 708 \
--load_GB_name ../save/GB-iccv.ckpt \
--gpu

python trainTextureTransfer.py \
--style_name ../data/style/${file1}.png \
--batchsize 2 --Ttraining_num 2 \
--texture_step1_epochs 40 \
--style_loss \
--save_path ../save --save_name ${file1} \
--text_path ../data/rawtext/yaheiB/train --text_datasize 708 \
--load_GS_name ../save/${file1}-GS.ckpt \
--gpu