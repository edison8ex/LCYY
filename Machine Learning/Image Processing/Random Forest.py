# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 16:27:46 2018

@author: Lcyy
"""

import numpy as np
import scipy.io
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import metrics
import timeit

def data_prep(uuu,kkk,aaa,bbb):
    target = scipy.io.loadmat(uuu)
    dataset = scipy.io.loadmat(kkk)
    haha = target[aaa][0]
    hehe = dataset[bbb]
    on = []
    for each in haha:
        on.append(each)
    hkust = []  
    for eachj in hehe:
        j = np.array_split(eachj,28)
        lcyy = []
        for each in j:
            lcyy.append(each.tolist())
        dse = np.asarray(lcyy)
        hkust.append(dse)
    comp = np.asarray(hkust)
    return on, comp

#Data preparation for training data
on = data_prep('train_labels.mat','train_images.mat','train_labels','train_images')[0]
stress = np.array(on)
comp = data_prep('train_labels.mat','train_images.mat','train_labels','train_images')[1]

#Data preparation for testing data
ggez = data_prep('test_labels.mat','test_images.mat','test_labels','test_images')[0]
cscs = data_prep('test_labels.mat','test_images.mat','test_labels','test_images')[1]
tension = np.array(ggez)

#running time measurement
start = timeit.default_timer()

#predoction on testing data set 
n_samples = len(comp)
data = comp.reshape((n_samples, -1))
m_samples = len(cscs)
dota = cscs.reshape((m_samples, -1))

#fitting the data
print('Fitting the dataset')
rfclf = RandomForestClassifier(random_state=0, max_features = None)
rfclf.fit(data, stress)

#training dataset accuracy
print('For training dataset')
rftrain_result = rfclf.predict(data)
print ('training dataset accuracy: {}'.format(accuracy_score(stress, rftrain_result)))

#confusion matrix + classification report
print("Classification report for classifier %s:\n%s\n"
      % (rfclf, metrics.classification_report(stress, rftrain_result)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(stress, rftrain_result))

#testing dataset accuracy
print('For testing dataset')
rf_result = rfclf.predict(dota)
print ('testing dataset accuracy: {}'.format(accuracy_score(tension, rf_result)))

#confusion matrix + classification report
print("Classification report for classifier %s:\n%s\n"
      % (rfclf, metrics.classification_report(tension, rf_result)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(tension, rf_result))

e1 = timeit.default_timer()
print ('Training time: {}'.format(e1-start))