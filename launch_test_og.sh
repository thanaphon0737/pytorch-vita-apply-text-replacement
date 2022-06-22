#!/bin/bash
filename='a'
model='holl2'
for filename in a a_d ad docq good o z
do
    python test.py \
    --text_name ../data/rawtext/yaheiB/val/test1a_data/inputTexture/${filename}.png \
    --scale -1 --scale_step 0.2 \
    --structure_model ../save/${model}-GS.ckpt \
    --texture_model ../save/${model}-GT.ckpt \
    --result_dir ../output/tr --name ${filename}-${model} \
    --gpu
done