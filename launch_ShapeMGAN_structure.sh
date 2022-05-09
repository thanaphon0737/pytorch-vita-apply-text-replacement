#!/bin/bash
python trainStructureTransfer.py \
--style_name ../data/style/1_f.png \
--batchsize 6 --Straining_num 1 \
--step1_epochs 30 --step2_epochs 40 --step3_epochs 80 \
--scale_num 4 \
--Sanglejitter \
--save_path ../save --save_name jess \
--text_path ../data/rawtext/yaheiB/train --text_datasize 708 \
--load_GB_name ../save/GB-iccv.ckpt \
--gpu