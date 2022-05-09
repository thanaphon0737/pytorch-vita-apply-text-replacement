#!/bin/bash
pic="0_sk_x"
model="10"
path='newtest_0_e_f_64-100ep-rec-100-sadv-0.1'

sname="newtest_0_e_f_skx_64-100ep-rec-100-sadv-0.1"
python test_s.py \
--text_name ../data/rawtext/yaheiB/val/test1a_data/${pic}.png \
--text_type 0 \
--scale 0 --scale_step 0.2 \
--structure_model ./predict/models/${path} \
--result_dir ../src/predict/${sname} --name ${sname} \
--gpu