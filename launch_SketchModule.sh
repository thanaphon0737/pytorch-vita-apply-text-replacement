#!/bin/bash
python trainSketchModule.py \
--text_path ../data/rawtext/yaheiB/train --text_datasize 708 \
--augment_text_path ../data/rawtext/augment --augment_text_datasize 5 \
--batchsize 4 --Btraining_num 1280 \
--save_GB_name ../save/GB.ckpt \
--gpu