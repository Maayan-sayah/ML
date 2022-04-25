import random
import sys

import numpy as np
from scipy.spatial.distance import cdist

def initilize(train_data,test_data,val_for_w):
    w = np.full((train_data.shape[1], 3),val_for_w)
    train_data = np.transpose(train_data)
    test_data = np.transpose(test_data)

    temp=[]
    for i in range(len(train_data[0])):
        temp.append(1)
    train_data = np.vstack((train_data, temp))
    temp = []
    for i in range(len(test_data[0])):
        temp.append(1)
    test_data = np.vstack((test_data, temp))
    temp=[]
    for i in range(len(w[0])):
        temp.append(1)
    w = np.vstack((w, temp))
    w=np.transpose(w)
    train_data = np.transpose(train_data)
    test_data = np.transpose(test_data)

    return train_data,test_data,w

def perceptron(train_data,train_lbl,test_data,epoc,eta):

    train_data, test_data, w=initilize(train_data, test_data,random.uniform(0, 1))
    predict = np.zeros((test_data.shape[0]))

    for e in range(epoc):
        c=list(zip(train_data,train_lbl))
        np.random.shuffle(c)
        for x,y in c:
            y=int(y)
            y_hat=np.argmax(np.dot(w,x))
            if y!=y_hat:
                w[y,:]=w[y,:]+eta*x
                w[y_hat, :]=w[y_hat,:]-eta*x

    for test in range(test_data.shape[0]):
        dotres=np.dot(w, test_data[test])
        mx = np.argmax(dotres)
        predict[test]=int(mx)

    return predict

def find_minmax(train_t):
    min_max_arr=[]
    for line in train_t:
        min=line.min()
        max=line.max()
        min_max_arr.append([min,max])
    return min_max_arr

def knn(train_data,train_lbl,test_data,k):

    distance_mtrix=cdist(test_data,train_data,'euclidean')
    tag_test_list=np.full((distance_mtrix.shape[0],), 0)

    for x in range(0,distance_mtrix.shape[0]):
        index_dic={}
        for i in range(0,distance_mtrix.shape[1]):
            index_dic[distance_mtrix[x][i]]=i
        distance_list=list(distance_mtrix[x])
        distance_list.sort()
        index_list=[]

        for dis in range(0,k):
            key=index_dic[distance_list[dis]]
            index_list.append(key)

        tag_list=[]
        for i in index_list:
            temp=train_lbl[i]
            tag_list.append(temp)

        tags=[0,0,0]

        for i in tag_list:
            tags[int(i)]+=1

        maxtag_indx=0

        for tag in range(len(tags)):
            if tags[tag]>tags[maxtag_indx]:
                maxtag_indx=tag

        tag_test_list[x]=int(maxtag_indx)
    return tag_test_list

def replace(w,y):
    temp=w.copy()
    for i in range(w.shape[0]):
        if i==y:
            temp[i]=np.full((len(w[i]),),0)

    return temp

def pa(train_data,train_lbl,test_data, i):
    train_data, test_data, w=initilize(train_data, test_data,0.0)
    predict = []

    for e in range(i):
        c = list(zip(train_data, train_lbl))
        np.random.shuffle(c)
        for x, y in c:
            y = int(y)
            temp = replace(w, y)

            y_hat = np.argmax(np.dot(temp, x))
            max_val = max(0.0, 1.0 - np.dot(w[y], x) + np.dot(w[y_hat], x))
            tau = max_val / (2 * ((np.linalg.norm(x)) ** 2))

            if y != y_hat:
                w[y, :] = w[y, :] + tau * x
                w[y_hat, :] = w[y_hat, :] - tau * x

    for test in test_data:
        mx = np.argmax(np.dot(w, test))
        predict.append(mx)

    return (predict)

def load_File(train_exmple_file_name,train_lable_file_name,test_file_name):

    train_lbl=np.loadtxt(train_lable_file_name,delimiter="/n")
    train_data=np.loadtxt(train_exmple_file_name,delimiter=",",
                          converters={11:lambda s:1 if s==b'W' else 0})
    test_data=np.loadtxt(test_file_name,delimiter=",",
                         converters={11:lambda s:1 if s==b'W' else 0})
    return train_lbl,train_data,test_data

def normlize(test_data,train_data):
    train_data = np.transpose(train_data)
    test_data = np.transpose(test_data)
    min_max_arr = find_minmax(train_data)

    for i in range(0,test_data.shape[0]) :
        min_n = min_max_arr[i][0]
        max_n = min_max_arr[i][1]
        for j in range(0,test_data.shape[1]):
            test_data[i,j] = (test_data[i,j] - min_n) / \
                          (max_n - min_n)

    for i in range(0,train_data.shape[0]) :
        min_n = min_max_arr[i][0]
        max_n = min_max_arr[i][1]
        for j in range(0,train_data.shape[1]):
            train_data[i,j] = (train_data[i,j] - min_n) / \
                          (max_n - min_n)

    train_data = np.transpose(train_data)
    test_data = np.transpose(test_data)

    return test_data,train_data

def main():
    train_lbl,train_data,test_data=load_File(sys.argv[1], sys.argv[2], sys.argv[3])
    test_data, train_data=normlize(test_data,train_data)
    knn_res= knn(train_data,train_lbl,test_data,8)
    perc_res = perceptron(train_data,train_lbl,test_data, 76, 0.5)
    pa_res = pa(train_data,train_lbl,test_data, 65)

    for i in range(len(test_data)):
        print(f"knn: {knn_res[i]}, perceptron: {int(perc_res[i])}, pa: {pa_res[i]}")


if __name__ == "__main__":
    main()



