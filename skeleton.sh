#!/bin/bash
for pic in  a d ad a_d o o3 o3s good codq z 0_std_BroadwayRegular
do
    python sk_trans4.py \
    --ref 0_BroadwayRegular_dis \
    --pic ${pic} 
done
