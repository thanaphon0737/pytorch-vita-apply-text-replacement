#!/bin/bash
pic="a_35_1_0_BroadwayRegular_dis_sk4"
model="137"
# n='docq_thin_dis'
# path=`newtest_${n}_64-100ep-rec-100-sadv-0.1`
sname='sk4_0_BroadwayRegular_35_1_0_BroadwayRegular_dis_sk4_5-240-3000ep-rec-100-sadv-0.1'
python test_one.py \
--text_name ../data/rawtext/yaheiB/val/test1a_data/k_35/${pic}.png \
--text_type 0 \
--pic ${pic} \
--scale 0 --scale_step 0.2 \
--structure_model ${sname} \
--result_dir ../output/testov/ --name ${sname} \
--gpu \
--model_n ${model}