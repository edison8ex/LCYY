# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 10:54:03 2018

@author: Lcyy
"""
import scipy.io
import timeit
from random import randint
import random
import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn import metrics
import matplotlib.pyplot as plt

def edist(datapoint,cen):
    edist = []
    for eachr in datapoint:
        edist.append(distance.euclidean(eachr, cen))    
    #print(len(edist))
    return edist

def ssecal(arr,ccdf,eeach,lcyy):
    KK = eeach
    ssss = []
    jeja = []
    while KK >= 1:
        ssee = 0
        gg = 0
        #sse part
        llx, lly = [], []
        eedf = ccdf.loc[ccdf['label_{}'.format(eeach)] == '{}'.format(KK)]
        eedff = eedf.reset_index(drop=False)
        iddxx = eedff['index']
        iddxx = iddxx.values
        centroidsse = (arr[2*KK-2],arr[2*KK-1])
        ubs = edist(lcyy[iddxx],centroidsse)
        distsselist = [d*d for d in ubs]
        ssec = sum(distsselist)
        ssee += ssec
        ssss.append(ssee)
        #new centorid part
        if len(iddxx) == 0:
            ran = 0
            while ran < 10:
                iddxx = np.append(iddxx,randint(0,499))
                ran+=1
        while gg<len(lcyy[iddxx]):
            llx.append(lcyy[iddxx][0][0])
            lly.append(lcyy[iddxx][0][1])
            gg += 1
        jeja.append(sum(llx)/len(llx))
        jeja.append(sum(lly)/len(lly))
        KK -= 1
        eedff.drop(['index'],axis=1)
    return ssss,jeja

def ploting(label,w):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(cdf['X-coor'],cdf['Y-coor'],c=cdf[label],s=50)
    ax.set_title('K-Means Clustering result for K={}'.format(w))
    plt.colorbar(scatter)
    
maxtrial = 20

cdf = pd.DataFrame(columns = ['X-coor','Y-coor'])
ddf = pd.DataFrame(columns = ['label'])
x, y = [], []
cluster = scipy.io.loadmat('dataset.mat')
for eachr in cluster['Points']:
    x.append(eachr[0])
    y.append(eachr[1])
cdf['X-coor'] = x
cdf['Y-coor'] = y
result,sil,ttt = [],[],[]

#running time measurement
start = timeit.default_timer()

for each in [2, 10, 20, 30]:
    t1 = timeit.default_timer()
    trial,itera = 1,2
    #gen initial centroid
    initialc = []
    ssselist = []
    times = each
    while times>=1:
        initialc.append(random.choice(x))
        initialc.append(random.choice(y))
        times-=1
    while True:
        if trial == 1:
            K = each
            while K >= 1:
                centroid = (initialc[2*K-2],initialc[2*K-1])
                dist = edist(cluster['Points'],centroid)
                cname = '{}'.format(K)
                ddf[cname] = dist
                K -= 1
            cdf['label_{}'.format(each)] = ddf.iloc[:,1:].apply(lambda x: x.idxmin(), axis=1)
            #calculate sse
            sselist = ssecal(initialc,cdf,each,cluster['Points'])[0]
            sse2 = sum(sselist)
            iterc = ssecal(initialc,cdf,each,cluster['Points'])[1]
            print('SSE for K =',each,'for iteration', '1' ,'is,',sse2)
        else:
            B = each
            while B>=1:
                ckk = (iterc[2*B-2],iterc[2*B-1])
                dist = edist(cluster['Points'],ckk)
                cname = '{}'.format(B)
                ddf[cname] = dist
                B -= 1
            cdf['label_{}'.format(each)] = ddf.iloc[:,1:].apply(lambda x: x.idxmin(), axis=1)
            sselist = ssecal(iterc,cdf,each,cluster['Points'])[0]
            sse2 = sum(sselist)
            iterc = ssecal(iterc,cdf,each,cluster['Points'])[1]
            print('SSE for K =',each,'for iteration', itera ,'is,',sse2)
            itera += 1
        ssselist.append(sse2)
        #print(mselist)
        if trial != 1:
            ssedelta = ssselist[trial-1]/ssselist[trial-2]-1
            sssedelta = ssselist[trial-2]/ssselist[trial-3]-1
            if (abs(ssedelta)<=0.001 and abs(sssedelta) <= 0.001) or trial >= maxtrial:
                break
        trial += 1 
    sil.append(metrics.silhouette_score(cluster['Points'], cdf['label_{}'.format(each)]))
    result.append(ssselist[len(ssselist)-1])
    t2 = timeit.default_timer()
    ttt.append(t2-t1)
    
print('\n')
pcdf = cdf.reset_index(drop=False)
print(pcdf)
df2 = pcdf.drop(columns=['label_10','label_20','label_30'])
df10 = pcdf.drop(columns=['label_2','label_20','label_30'])
df20 = pcdf.drop(columns=['label_10','label_2','label_30'])
df30 = pcdf.drop(columns=['label_10','label_2','label_20'])
np.savetxt('K2label.txt', df2.values, fmt = '%s', delimiter=",\t", header="PointID\tX-Coordinate\tY-Coordinate\tClusterID") 
np.savetxt('K10label.txt', df10.values, fmt = '%s', delimiter=",\t", header="PointID\tX-Coordinate\tY-Coordinate\tClusterID") 
np.savetxt('K20label.txt', df20.values, fmt = '%s', delimiter=",\t", header="PointID\tX-Coordinate\tY-Coordinate\tClusterID") 
np.savetxt('K30label.txt', df30.values, fmt = '%s', delimiter=",\t", header="PointID\tX-Coordinate\tY-Coordinate\tClusterID") 
print('\n')
print('SSE for K =2 is', result[0])
print('Silhouette Coefficient:',sil[0])
print ('Training time for K = {} is '.format(each),ttt[0])
ploting('label_2',2)
print('SSE for K =10 is', result[1])
print('Silhouette Coefficient:',sil[1])
print ('Training time for K = {} is '.format(each),ttt[1])
ploting('label_10',10)
print('SSE for K =20 is', result[2])
print('Silhouette Coefficient:',sil[2])
print ('Training time for K = {} is '.format(each),ttt[2])
ploting('label_20',20)
print('SSE for K =30 is', result[3])
print('Silhouette Coefficient:',sil[3])
print ('Training time for K = {} is '.format(each),ttt[3])
ploting('label_30',30)

e1 = timeit.default_timer()
print ('Training time: {}'.format(e1-start))