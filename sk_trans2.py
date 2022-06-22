import cv2
import numpy as np
from utils import text_image_preprocessing, save_image
from PIL import Image
import os
name = '0_BAUHS93'
filepath = './newset/var_fonts/' + name + '.png'
savepath = './newset/var_fonts/' + name
img = cv2.imread(filepath,0)



h_s = 256

size = img.shape
newsize = size
print(size[0],size[1])
h = size[0]
w = size[1]
if(h <= 256):
    ratio = (int(h_s/h) +1)
    ratio = (round(ratio/4)+1) * 4
    newsize =(w*ratio,h*ratio)
    img = cv2.resize(img, newsize, interpolation = cv2.INTER_AREA)

# print(filepath.split('_'))
# if 'dis.png' not in filepath.split('_'):
#     img = text_image_preprocessing(filepath,newsize,False)
#     img.save(savepath +'_dis' +'.png')
#     img = cv2.imread(savepath +'_dis' +'.png')
# (B,G,R) = cv2.split(img)



# skeletion process in R channel
ret,img = cv2.threshold(img,127,255,0)
skel = np.zeros(img.shape, np.uint8)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
while True:
    open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
    temp = cv2.subtract(img,open)
    eroded = cv2.erode(img,element)
    skel = cv2.bitwise_or(skel,temp)
    img = eroded.copy()
    if cv2.countNonZero(img) == 0:
        break
# merge all channel 
B = np.ones(img.shape,np.uint8)
B = B*255
G = np.zeros(img.shape,np.uint8)
print(np.max(skel))
merged = cv2.merge([B,G,skel])
cv2.imshow('skel',merged)
cv2.imshow('img',img)

if 'sk_for_train' not in os.listdir('./newset'):
    os.makedirs('./newset/sk_for_train')
cv2.imwrite('./newset/sk_for_train/'+ name +'_sk3' '.png',merged)

# print('----save----')
cv2.waitKey(0)