import os
# print(os.listdir('../data/rawtext/yaheiB/val/test1a_data/k_55'))
l = os.listdir('../data/rawtext/yaheiB/val/test1a_data/k_53')
with open('listfile.txt', 'w') as f:
    for fi in l:
        print(fi.split('.'))
        f.write(str(fi.split('.')[0]))
        f.write(' ')