# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 21:07:50 2018

@author: Lcyy
"""
#Few of the lines in DBSCAN scan code might be similar to few other RMBI students
#It is because these codes were RMBI3110 assignment sample codes
#Credit to my TA in RMBI3110, Mina

import scipy.io
import timeit
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

cdf = pd.DataFrame(columns = ['X-coor','Y-coor'])
ddf = pd.DataFrame(columns = ['label'])
x, y = [], []
cluster = scipy.io.loadmat('dataset.mat')
for eachr in cluster['Points']:
    x.append(eachr[0])
    y.append(eachr[1])
cdf['X-coor'] = x
cdf['Y-coor'] = y
feature_vector = ['X-coor','Y-coor']

#running time measurement
start = timeit.default_timer()

db = DBSCAN(eps=0.12,min_samples=3).fit(cluster['Points'])
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

cdf['label'] = [c+1 for c in labels]
print(cdf)
    
# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
plt.figure(figsize=(15, 10))

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)

    xy = cluster['Points'][class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)

    xy = cluster['Points'][class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=6)
    
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

pcdf = cdf.reset_index(drop=False)
np.savetxt('DBSCANlabel.txt', pcdf.values, fmt = '%s', delimiter=",\t", header="PointID\tX-Coordinate\tY-Coordinate\tClusterID") 

e1 = timeit.default_timer()
print ('Training time: {}'.format(e1-start))
