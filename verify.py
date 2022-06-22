import pickle
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import cv2
import os
import csv
from options import Trainandverify



# cv2.imshow('img',img)
# cv2.waitKey(0)
C = './predict'
D = 'D:\work\Masterdegree\shapmatching\predict'
parser = Trainandverify()
opts = parser.parse()
print(opts.file_name)
path = C
file_list = os.listdir(path)
filter_list = []
target = opts.file_name
img = cv2.imread('../data/style/' + opts.pic+'.png').astype('float')
h,w,_ = img.shape
h = int(h)
w = int(w)
for f in file_list:
    if target in f :
        filter_list.append(f)

for fi in filter_list: 

    # file_target = '0_e_f_6_19042022-1250im-10001ep-rec-100-sadv-4.0'
    file_target = fi
    arr = os.listdir(os.path.join(path , file_target))
    b = dict()
    g = dict()
    r = dict()
    t = dict()
    length = len(arr)
    i =0
    for filename in arr:
        img2 = cv2.imread(os.path.join(os.path.join(path , file_target),filename)).astype('float')
        nh,nw,_ = img2.shape
        img = cv2.resize(img,(nw,nh))
        b.update({int(filename.split('_')[0]):np.mean(np.abs(img[:,:,0]-img2[:,:,0])).astype('float')})
        g.update({int(filename.split('_')[0]):np.mean(np.abs(img[:,:,1]-img2[:,:,1])).astype('float')})
        r.update({int(filename.split('_')[0]):np.mean(np.abs(img[:,:,2]-img2[:,:,2])).astype('float')})
        t.update({int(filename.split('_')[0]):np.mean(np.abs(img-img2)).astype('float')})
        i += 1
        print('Process: %d/%d'%(i,length))

    s_b = dict(sorted(b.items()))
    s_g = dict(sorted(g.items()))
    s_r = dict(sorted(r.items()))
    s_t = dict(sorted(t.items()))
    epoch = []
    b_err = []
    g_err = []
    r_err = []
    t_err = []
    for k,v in s_b.items():
        
        epoch.append(k)
        b_err.append(v)

    for k,v in s_g.items():
        g_err.append(v)

    for k,v in s_r.items():
        r_err.append(v)
    for k,v in s_t.items():
        t_err.append(v)

    file = open('./losses/' + file_target +'.csv')
    csvreader = csv.reader(file)
    rows=[]
    for row in csvreader:
        
        if row[3] != 'Lrec':
                rows.append(row[3])
    file.close

    save_list = {
        'epoch': epoch,
        'B_err': b_err,
        'G_err': g_err,
        'R_err': r_err,
        'Total_err': t_err,
        'Train_err': rows,
    }

    df = pd.DataFrame(save_list)
    df.to_csv(r'../src/ver_losses/' + file_target + '.csv',index=False)
# img3 = (img-img2)

# print(np.mean(img3))

