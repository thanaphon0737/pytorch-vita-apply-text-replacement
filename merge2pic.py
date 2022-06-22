import cv2
import os
import numpy as np
route = '../output/testov'
files = os.listdir(route)
for fil in files:
    if fil != 'output':
        path = os.path.join(route,fil)
        l = os.listdir(path)
        img_stack = []
        for fi in l:
            # print(fi)
            img_stack.append(cv2.imread(os.path.join(path,fi)))
        cv2.imshow('im1',img_stack[0])
        cv2.imshow('im2',img_stack[1])
        im1 = img_stack[0].astype(float)
        im2 = img_stack[1].astype(float)
        # plus and divide by 2
        im3 = ((img_stack[0].astype(float) + img_stack[1].astype(float))/2)
        im3 = cv2.normalize(im3, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # max
        im4 = np.maximum(im1,im2)
        im4 = cv2.normalize(im4, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    
        save = '../output/testov/output'
        cv2.imwrite(os.path.join(save,fil + '-im3.png'),im3)
        cv2.imwrite(os.path.join(save,fil + '-im4.png'),im4)