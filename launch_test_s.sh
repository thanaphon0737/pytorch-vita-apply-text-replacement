#!/bin/bash
# pic="o3_0_BroadwayRegular_dis_sk4"
model="352"
# n='docq_thin_dis'
# path=`newtest_${n}_64-100ep-rec-100-sadv-0.1`
sname='sk4_0_BroadwayRegular_77_1_0_BroadwayRegular_dis_sk4_3-204-3000ep-rec-100-sadv-0.1'
sub='204'
k='77'
# a_33_0_BroadwayRegular_dis_sk4 a_d_33_0_BroadwayRegular_dis_sk4 ad_33_0_BroadwayRegular_dis_sk4 codq_33_0_BroadwayRegular_dis_sk4 d_33_0_BroadwayRegular_dis_sk4 good_33_0_BroadwayRegular_dis_sk4 o_33_0_BroadwayRegular_dis_sk4 o3_33_0_BroadwayRegular_dis_sk4 o3s_33_0_BroadwayRegular_dis_sk4 z_33_0_BroadwayRegular_dis_sk4
# a_53_0_BroadwayRegular_dis_sk4 a_d_53_0_BroadwayRegular_dis_sk4 ad_53_0_BroadwayRegular_dis_sk4 codq_53_0_BroadwayRegular_dis_sk4 d_53_0_BroadwayRegular_dis_sk4 good_53_0_BroadwayRegular_dis_sk4 o_53_0_BroadwayRegular_dis_sk4 o3_53_0_BroadwayRegular_dis_sk4 o3s_53_0_BroadwayRegular_dis_sk4 z_53_0_BroadwayRegular_dis_sk4
# a_35_0_BroadwayRegular_dis_sk4 a_d_35_0_BroadwayRegular_dis_sk4 ad_35_0_BroadwayRegular_dis_sk4 codq_35_0_BroadwayRegular_dis_sk4 d_35_0_BroadwayRegular_dis_sk4 good_35_0_BroadwayRegular_dis_sk4 o_35_0_BroadwayRegular_dis_sk4 o3_35_0_BroadwayRegular_dis_sk4 o3s_35_0_BroadwayRegular_dis_sk4 z_35_0_BroadwayRegular_dis_sk4 0_std_BroadwayRegular_35_0_BroadwayRegular_dis_sk4
for pic in 0_BroadwayRegular_${k}_1_0_BroadwayRegular_dis_sk4 0_std_BroadwayRegular_${k}_1_0_BroadwayRegular_dis_sk4 ad_${k}_1_0_BroadwayRegular_dis_sk4 a_${k}_1_0_BroadwayRegular_dis_sk4 a_d_${k}_1_0_BroadwayRegular_dis_sk4 codq_${k}_1_0_BroadwayRegular_dis_sk4 d_${k}_1_0_BroadwayRegular_dis_sk4 good_${k}_1_0_BroadwayRegular_dis_sk4 o3s_${k}_1_0_BroadwayRegular_dis_sk4 o3_${k}_1_0_BroadwayRegular_dis_sk4 o_${k}_1_0_BroadwayRegular_dis_sk4 z_${k}_1_0_BroadwayRegular_dis_sk4
do
    python test_s.py \
    --text_name ../data/rawtext/yaheiB/val/test1a_data/k_${k}/${pic}.png \
    --text_type 0 \
    --pic ${pic} \
    --scale 0 --scale_step 0.2 \
    --structure_model ${sname} \
    --result_dir ../output/test1a/BroadWay_${k}_${sub}_${sub} --name ${sname} \
    --gpu \
    --model_n ${model} \
    --sub_image ${sub} 
done