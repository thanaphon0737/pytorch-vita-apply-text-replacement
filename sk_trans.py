import cv2
import numpy as np
from utils import text_image_preprocessing, save_image
from PIL import Image
import os
name = '0_SHOWG'
filepath = './newset/var_fonts/' + name + '.png'
savepath = './newset/var_fonts/' + name
img = cv2.imread(filepath)



h_s = 256

size = img.shape
newsize = 0
print(size[0],size[1])
h = size[0]
w = size[1]
if(h <= 256):
    ratio = (int(h_s/h) +1)
    ratio = (round(ratio/4)+1) * 4
    newsize =(w*ratio,h*ratio)
    img = cv2.resize(img, newsize, interpolation = cv2.INTER_AREA)

print(filepath.split('_'))
if 'dis.png' not in filepath.split('_'):
    img = text_image_preprocessing(filepath,newsize,False)
    img.save(savepath +'_dis' +'.png')
    img = cv2.imread(savepath +'_dis' +'.png')
(B,G,R) = cv2.split(img)
ret,R = cv2.threshold(R,127,255,0)


# skeletion process in R channel
sk_size = np.size(R)
skel = np.zeros(R.shape, np.uint8)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
while True:
    open = cv2.morphologyEx(R, cv2.MORPH_OPEN, element)
    temp = cv2.subtract(R,open)
    eroded = cv2.erode(R,element)
    skel = cv2.bitwise_or(skel,temp)
    R = eroded.copy()
    if cv2.countNonZero(R) == 0:
        break
# merge all channel 

merged = cv2.merge([B,G,skel])
cv2.imshow('skel',merged)
cv2.imshow('img',img)
if 'sk_for_train' not in os.listdir('./newset'):
    os.makedirs('./newset/sk_for_train')
cv2.imwrite('./newset/sk_for_train/'+ name +'_sk' '.png',merged)
# print('----save----')
cv2.waitKey(0)