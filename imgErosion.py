from utils import text_image_preprocessing, save_image
import cv2
import numpy as np
from PIL import Image
filepath = './newset/'
savepath = './newset/result3/'
i='1_f'
h_s = 256
# Reading the input image
img = cv2.imread(filepath +str(i) +'.png', 0)
size = img.shape[1],img.shape[0]
img1 = Image.open(filepath + str(i)+'.png')
w,h = img1.size
ratio = (int(h_s/h) +1)
ratio = (round(ratio/4)+1) * 4
newsize =(w*ratio,h*ratio)

# Taking a matrix of size 5 as the kernel
kernel = np.ones((20,20), np.uint8)
img = cv2.resize(img,newsize)
# The first parameter is the original image,
# kernel is the matrix with which image is
# convolved and third parameter is the number
# of iterations, which will determine how much
# you want to erode/dilate a given image.
img_erosion = cv2.erode(img, kernel, iterations=1)
img_erosion = cv2.resize(img_erosion,size)
print(size)
cv2.imshow('Input', img)
cv2.imshow('Erosion', img_erosion)

cv2.imwrite(savepath + i+'.png',img_erosion)
cv2.waitKey(0)
