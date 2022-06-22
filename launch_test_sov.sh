#!/bin/bash
model="137"
sname='sk4_0_BroadwayRegular_35_1_0_BroadwayRegular_dis_sk4_5-240-3000ep-rec-100-sadv-0.1'
sub='204'
k='35'
pic='a_d_35_1_0_BroadwayRegular_dis_sk4'
A='100'
for pic in 0_std_BroadwayRegular_35_1_0_BroadwayRegular_dis_sk4 ad_35_1_0_BroadwayRegular_dis_sk4 a_35_1_0_BroadwayRegular_dis_sk4 a_d_35_1_0_BroadwayRegular_dis_sk4 codq_35_1_0_BroadwayRegular_dis_sk4 d_35_1_0_BroadwayRegular_dis_sk4 good_35_1_0_BroadwayRegular_dis_sk4 o3s_35_1_0_BroadwayRegular_dis_sk4 o3_35_1_0_BroadwayRegular_dis_sk4 o_35_1_0_BroadwayRegular_dis_sk4 z_35_1_0_BroadwayRegular_dis_sk4 
do
    python test_sov.py \
        --text_name ../data/rawtext/yaheiB/val/test1a_data/k_${k}/${pic}.png \
        --text_type 0 \
        --pic ${pic} \
        --scale 0 --scale_step 0.2 \
        --structure_model ${sname} \
        --result_dir ../output/testov/${pic}_${k}_${sub}_${sub} --name ${sname} \
        --gpu \
        --model_n ${model} \
        --sub_image ${sub} \
        --A ${A}
    python test_sov_basic.py \
        --text_name ../data/rawtext/yaheiB/val/test1a_data/k_${k}/${pic}.png \
        --text_type 0 \
        --pic ${pic} \
        --scale 0 --scale_step 0.2 \
        --structure_model ${sname} \
        --result_dir ../output/testov/${pic}_${k}_${sub}_${sub} --name ${sname} \
        --gpu \
        --model_n ${model} \
        --sub_image ${sub} 
done