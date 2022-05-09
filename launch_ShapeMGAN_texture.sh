#!/bin/bash
python trainTextureTransfer.py \
--style_name ../data/style/blue.png \
--batchsize 2 --Ttraining_num 150 \
--texture_step1_epochs 40 \
--style_loss \
--save_path ../save --save_name blue \
--text_path ../data/rawtext/yaheiB/train --text_datasize 708 \
--load_GS_name ../save/blue-GS.ckpt \
--gpu