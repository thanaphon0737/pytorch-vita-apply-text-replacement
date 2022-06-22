import pandas as pd
import csv
import numpy as np
import os
from options import ClearFile
parser = ClearFile()
opts = parser.parse()
def main():
    file_target = opts.file_name
    file = open('./ver_losses/' + file_target +'.csv')
    csvreader = csv.reader(file)
    rows=[]
    ep = []

    for row in csvreader:
        if 'epoch' not in row[0][:]:
                # print(type(row[4]))
                rows.append((float(row[4])))
                ep.append(row[0])
    print(rows)
    a = zip(rows,ep)
    min_list = list(min(a))
    min_ep = min_list[1]
    min_loss = min_list[0]
    print(min_ep,min_loss)
    print('confirm y/n:')
    # x = input()
    # if x == 'Y' or x == 'y':
    if True:
        
        folder_pic = 'D:\work\Masterdegree\shapmatching\predict'
        folder_model = 'D:\work\Masterdegree\shapmatching\predict\models'
        folder_pic = './predict'
        folder_model = './predict/models'
        path = os.path.join(folder_pic,file_target)
        model_path = os.path.join(folder_model,file_target)
        print(os.listdir(path))
        print(os.listdir(model_path))
        i = 0
        all_file = len(os.listdir(path))
        for fi in os.listdir(path):
            if fi.split('_')[0] != min_ep:
                myfile=os.path.join(path,fi)

                ## Try to delete the file ##
                try:
                    os.remove(myfile)
                    print('Remove :',fi)
                except OSError as e:  ## if failed, report it back to the user ##
                    print ("Error: %s - %s." % (e.filename, e.strerror))
                i += 1
            
            print('Process: %d/%d'%(i,all_file))
        j = 0
        all_file_m = len(os.listdir(model_path))
        for m in os.listdir(model_path):
            if m.split('-')[0] != min_ep:
                myfile=os.path.join(model_path,m)

                ## Try to delete the file ##
                try:
                    os.remove(myfile)
                    print('Remove :',fi)
                except OSError as e:  ## if failed, report it back to the user ##
                    print ("Error: %s - %s." % (e.filename, e.strerror))
                j += 1
            print('Process: %d/%d'%(j,all_file_m))
    else:
        print('cancel')


    file.close

main()

