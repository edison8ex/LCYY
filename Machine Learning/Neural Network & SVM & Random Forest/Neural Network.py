# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 22:55:44 2018

@author: Lcyy
"""
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import scipy.io
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
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

def CVNN(num, ds, dl):
    #k-fold cross validation to determine # of hidden unit
    kf = KFold(n_splits=5) # 5 fold
    classifier = MLPClassifier(hidden_layer_sizes=(num,), random_state=1)
    meancv = []
    fold = 1
    print(num, ' hidden unit:')
    for train_sector, test_sector in kf.split(ds):
        ds_train, ds_test = ds[train_sector], ds[test_sector]
        dl_train, dl_test = dl[train_sector], dl[test_sector]
        cfr = classifier.fit(ds_train,dl_train)
        nn_result = cfr.predict(ds_test)
        meancv.append(accuracy_score(dl_test, nn_result))
        print ('accuracy for fold ' ,fold, ' : {}'.format(accuracy_score(dl_test, nn_result)))
        fold += 1
    print('Mean accuracy for ', num, ' hidden unit in 5-fold cross validation:', sum(meancv)/len(meancv))  

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

#perform cross-validation to determine # of hidden unit
print('Perform cross-validation')
for numhiddenunit in [1,5,10,20,50]:
    CVNN(numhiddenunit, data, stress)
print('When H equals 50, cross validation sets gives the best mean accuracy')

#fitting the data
print('Fitting the dataset')
nnclf = MLPClassifier(hidden_layer_sizes=50)
nnclf.fit(data, stress)

#training dataset accuracy
print('For training dataset')
nntrain_result = nnclf.predict(data)
print ('training dataset accuracy: {}'.format(accuracy_score(stress, nntrain_result)))

#confusion matrix + classification report
print("Classification report for classifier %s:\n%s\n"
      % (nnclf, metrics.classification_report(stress, nntrain_result)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(stress, nntrain_result))

#testing dataset accuracy
print('For testing dataset')
nnet_result = nnclf.predict(dota)
print ('testing dataset accuracy: {}'.format(accuracy_score(tension, nnet_result)))

#confusion matrix + classification report
print("Classification report for classifier %s:\n%s\n"
      % (nnclf, metrics.classification_report(tension, nnet_result)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(tension, nnet_result))

e1 = timeit.default_timer()
print ('Training time: {}'.format(e1-start))
