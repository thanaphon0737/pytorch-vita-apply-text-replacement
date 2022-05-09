import pickle
import matplotlib.pyplot as plt 
import numpy
import pandas as pd
 
# reading csv file
# pd.read_csv("0_e_f_6_28032022.csv")
import csv
fname = '0_e_f_6_19042022-1250im-10001ep-rec-100-sadv-4.0'
file = open('./losses/'+fname + '.csv')
csvreader = csv.reader(file)
header = next(csvreader)
print(header)
rows = []
for row in csvreader:
    rows.append(row)
# print(rows)
# print(type(rows[0][1]))
file.close()

arr = numpy.array(rows)
# print(arr)
# x = numpy.arange(0,10000)
y1 = []
y2 = []
y3 = []
temp1 = arr[:,1]
temp2 = arr[:,2]
temp3 = arr[:,3]
for i in range(len(arr[:,1])):
    # if i%1000 == 0:
    y1.append(temp1[i])
    y2.append(temp2[i])
    y3.append(temp3[i])
temp1 = numpy.array(temp1).astype(numpy.float64)
temp2 = numpy.array(temp2).astype(numpy.float64)
temp3 = numpy.array(temp3).astype(numpy.float64)
loss_total = temp1 + temp2 + temp3
print(type(numpy.array(y1)),y2,y3)
plt.title('loss_adv' + fname )
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(numpy.array(y1).astype(numpy.float64))
plt.plot(numpy.array(y2).astype(numpy.float64))
# plt.plot(x,numpy.array(y3).astype(numpy.float))
plt.legend(['LGadv','LDadv','Lrec'],loc='upper left')
plt.savefig('./predict/' + fname +'/' + 'loss-adv' +'.png')
plt.show()

plt.title('loss_rec' +fname )
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(numpy.array(y3).astype(numpy.float64))
plt.legend(['LGadv','LDadv','Lrec'],loc='upper left')
plt.savefig('./predict/' + fname +'/' + 'loss-rec' +'.png')
plt.show()

plt.title('loss_total' +fname )
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(loss_total)
plt.legend(['total'],loc='upper left')
plt.savefig('./predict/' + fname +'/' + 'loss-total' +'.png')
plt.show()
