import cv2
import numpy as np
from utils import text_image_preprocessing, save_image
from PIL import Image
import os
from options import Skeleton
parser = Skeleton()
opts = parser.parse()
def resizefix_h(h,w,h_o,w_o):
    r = h/h_o
    print(h,h_o)
    return (int(w_o*r),int(h_o*r))
name = opts.pic
filepath = './newset/var_fonts/' + name + '.png'
savepath = './newset/var_fonts/' + name
img = cv2.imread(filepath,0)
ref = opts.ref
img_ref = cv2.imread('./newset/var_fonts/' + ref + '.png')
# print(img_ref)
s = img_ref.shape
h = s[0]
w = s[1]

size = img.shape
newsize = size
# print(size[0],size[1])
h_o = size[0]
w_o = size[1]
if(h_o <= 256):
    # ratio = (int(h_s/h) +1)
    # ratio = (round(ratio/4)+1) * 4
    # newsize =(w*ratio,h*ratio)
    newsize = resizefix_h(h,w,h_o,w_o)
    img = cv2.resize(img, newsize, interpolation = cv2.INTER_AREA)
# h,w = img.shape
# fix_w = 624
# newsize = (fix_w,round((h/w)*fix_w))
# img = cv2.resize(img, newsize, interpolation = cv2.INTER_AREA)
# print(filepath.split('_'))
# if 'dis.png' not in filepath.split('_'):
#     img = text_image_preprocessing(filepath,newsize,False)
#     img.save(savepath +'_dis' +'.png')
#     img = cv2.imread(savepath +'_dis' +'.png')
# (B,G,R) = cv2.split(img)


kernel = (7,7)
# skeletion process in R channel
ret,img = cv2.threshold(img,127,255,0)
skel = np.zeros(img.shape, np.uint8)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, kernel)
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
B = B*0
G = np.zeros(img.shape,np.uint8)

merged = cv2.merge([B,G,skel])
gray = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)
# cv2.imshow('skel',merged)
# cv2.imshow('img',img)
ret,thresh_img = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
# cv2.imshow('therh',thresh_img)
imgs,contours,_ = cv2.findContours(thresh_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
# draw the contours on the empty image
# cv2.drawContours(img_contours, contour, -1, (0,255,0), 3)
bb_out = merged.copy()
l_noise = []
for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        # default 300
        if w*h < 300:
            l_noise.append([x,y,w,h])
            cv2.rectangle(bb_out, (x, y), (x + w, y + h), (255, 255, 255), 2)
# print(contours)
# print(merged.shape)
for x,y,w,h in l_noise:
    merged[y:y+h,x:x+w] = 0
# print(np.max(merged[:,:,0]))
# print(np.max(merged[:,:,1]))
# print(np.max(merged[:,:,2]))
# cv2.imshow('gray',bb_out)
# cv2.imshow('out',merged)

version = '1'
file_save = 'sk_for_train_' + str(kernel[0]) + str(kernel[1]) + '_' + version
if file_save not in os.listdir('./newset'):
    os.makedirs('./newset/' + file_save)
cv2.imwrite('./newset/' + file_save + '/'+ name+'_' + str(kernel[0]) + str(kernel[1]) + '_' +version + '_' +ref +'_sk4' '.png',merged)

print('----save----')
print('./newset/' + file_save + '/'+ name+'_' +str(kernel[0]) + str(kernel[1])+ '_' +version+ '_' +ref +'_sk4' '.png')
# cv2.waitKey(0)