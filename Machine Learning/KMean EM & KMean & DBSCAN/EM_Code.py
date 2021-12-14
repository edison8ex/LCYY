# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 10:54:03 2018

@author: Lcyy
"""
import scipy.io
import timeit
import random
import pandas as pd
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

def edist(datapoint,cen):
    edist = []
    for eachr in datapoint:
        edist.append(distance.euclidean(eachr, cen))    
    #print(len(edist))
    return edist

def ssecal(arr,ccdf,eeach,lcyy,ix,iy):
    KK = eeach
    ssss= []
    jeja = []
    ssss = 0
    #sse part
    eedf = ccdf.loc[ccdf['label_{}'.format(eeach)] == '{}'.format(KK)] #cluster2
    eedff = eedf.reset_index(drop=False)
    jpm = (arr[2*KK-2],arr[2*KK-1]) #c2
    ubs = edist(lcyy[:],jpm) #dist(o,c2)
    distto1stclist = [d*d for d in ubs] #dist(o,c2)^2
    ms = (arr[2*KK-4],arr[2*KK-3]) #c1
    cs = edist(lcyy[:],ms) #dist(o,c1)
    distto1stcl2nd = [dd*dd for dd in cs] #dist(o,c1)^2
    sumlist = [x+y for x,y in zip(distto1stclist,distto1stcl2nd)]
    ww2 = [a/b for a,b in zip(distto1stclist,sumlist)] #w2
    ww1 = [1-ca for ca in ww2] #w1
    #centroid part
    ww22 = [aa*aa for aa in ww2]
    ww12 = [bb*bb for bb in ww1]
    ssec2 = sum([m*n for m,n in zip(distto1stclist,ww22)])
    ssec1 = sum([mm*nn for mm,nn in zip(distto1stcl2nd,ww12)])
    ssss = ssec2+ssec1
    #centroid2
    wc2xlist = [f*h for f,h in zip(ww22,x)] 
    wc2ylist = [u*v for u,v in zip(ww22,y)] 
    llc2x = sum(wc2xlist)/sum(ww22)
    llc2y = sum(wc2ylist)/sum(ww22)
    #centorid2 
    wc1xlist = [f*h for f,h in zip(ww12,x)] 
    wc1ylist = [u*v for u,v in zip(ww12,y)]         
    llc1x = sum(wc1xlist)/sum(ww12)
    llc1y = sum(wc1ylist)/sum(ww12)
    #output
    jeja.extend([llc1x,llc1y,llc2x,llc2y])
    KK -= 1
    eedff.drop(['index'],axis=1)
    #print(jeja)
    return ssss,jeja

def ploting(label,w):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(cdf['X-coor'],cdf['Y-coor'],c=cdf[label],s=50)
    ax.set_title('K-Means EM Clustering result for K={}'.format(w))
    plt.colorbar(scatter)

maxtrial = 50

cdf = pd.DataFrame(columns = ['X-coor','Y-coor'])
ddf = pd.DataFrame(columns = ['label'])
x, y = [], []
cluster = scipy.io.loadmat('dataset.mat')
for eachr in cluster['Points']:
    x.append(eachr[0])
    y.append(eachr[1])
cdf['X-coor'] = x
cdf['Y-coor'] = y
result = []

#running time measurement
start = timeit.default_timer()

for each in [2]:
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
            sse2 = ssecal(initialc,cdf,each,cluster['Points'],x,y)[0]
            iterc = ssecal(initialc,cdf,each,cluster['Points'],x,y)[1]
            print('iteration 1:')
            print('SSE for K =',each,'is,',sse2)
            print('Center for cluster 1 is: ', iterc[0],iterc[1])
            print('Center for cluster 2 is: ', iterc[2],iterc[3])
        else:
            B = each
            while B>=1:
                ckk = (iterc[2*B-2],iterc[2*B-1])
                dist = edist(cluster['Points'],ckk)
                cname = '{}'.format(B)
                ddf[cname] = dist
                B -= 1
            cdf['label_{}'.format(each)] = ddf.iloc[:,1:].apply(lambda x: x.idxmin(), axis=1)
            sse2 = ssecal(iterc,cdf,each,cluster['Points'],x,y)[0]
            iterc = ssecal(iterc,cdf,each,cluster['Points'],x,y)[1]
            print('iteration {}:'.format(itera))
            print('SSE for K =',each,'is,',sse2)
            print('Center for cluster 1 is: ', iterc[0],iterc[1])
            print('Center for cluster 2 is: ', iterc[2],iterc[3])
            itera += 1
        ssselist.append(sse2)
        #print(mselist)
        if trial != 1:
            ssedelta = ssselist[trial-1]/ssselist[trial-2]-1
            sssedelta = ssselist[trial-2]/ssselist[trial-3]-1
            if (abs(ssedelta)<=0.001 and abs(sssedelta) <= 0.001) or trial >= maxtrial:
                break
        trial += 1 
    result.append(ssselist[len(ssselist)-1])
    
print('\n')
print(cdf)
pcdf = cdf.reset_index(drop=False)
np.savetxt('K2EMlabel.txt', pcdf.values, fmt = '%s', delimiter=",\t", header="PointID\tX-Coordinate\tY-Coordinate\tClusterID") 
print('\n')
print('SSE for K =2 is', result[0])
ploting('label_2',2)

e1 = timeit.default_timer()
print ('Training time: {}'.format(e1-start))