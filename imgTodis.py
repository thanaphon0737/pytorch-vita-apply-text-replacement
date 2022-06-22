from utils import text_image_preprocessing, save_image
import cv2
import numpy as np
from PIL import Image
filepath = './newset/var_fonts/'
savepath = './newset/var_fonts/'
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

h_s = 256

i = 'BroadwayRegular_2_33'

img1 = Image.open(filepath + str(i)+'.png')
w,h = img1.size
print(w,h)
if(h <= 256):
    ratio = (int(h_s/h) +1)
    ratio = (round(ratio/4)+1) * 4
    newsize =(w*ratio,h*ratio)
    img1 = img1.resize(newsize)
    img2 = text_image_preprocessing(filepath + str(i)+'.png',newsize,False)
    # img2 = text_image_preprocessing(filepath + str(i)+'.png')
    img2.save(savepath + str(i) + '_dis' +'.png')
    print('save')
else:
    img2 = text_image_preprocessing(filepath + str(i)+'.png',False)
    img2.save(savepath + str(i) + '_dis' +'.png')
    print('save')