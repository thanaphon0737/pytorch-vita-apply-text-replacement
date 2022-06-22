from utils import text_image_preprocessing, save_image
import cv2
import numpy as np
from PIL import Image
import os
filepath = '../data/rawtext/yaheiB/val/test1a_data/fromSt'
savepath = '../data/rawtext/yaheiB/val/test1a_data/bw'
savepathfinal = '../data/rawtext/yaheiB/val/test1a_data/inputTexture'
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

# print(os.listdir(filepath))
l = os.listdir(filepath)
for fi in l:
    print(fi)
    img = cv2.imread(os.path.join(filepath,fi),0)
    
    ret,img = cv2.threshold(img,127,255,0)
    # cv2.imshow('out',img)
    print(os.path.join(filepath,fi))
    cv2.imwrite(os.path.join(savepath,fi),img)
# cv2.waitKey(0)
for fi in l:

    img2 = text_image_preprocessing(os.path.join(savepath,fi),None,False)
    img2.save(os.path.join(savepathfinal,fi))    

