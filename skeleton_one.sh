#!/bin/bash
for pic in 0_BroadwayRegular
do
    python sk_trans4.py \
    --ref 0_BroadwayRegular_dis \
    --pic ${pic} 
done
