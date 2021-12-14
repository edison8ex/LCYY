import numpy as np
import scipy.io
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import timeit
from sklearn import metrics

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

def CVSVM(gam, ds, dl):
    #k-fold cross validation to determine # of hidden unit
    kf = KFold(n_splits=5) # 5 fold
    svmm = svm.SVC(kernel='rbf',gamma=gam,random_state=1) 
    meansv = []
    fold = 1
    print('Gamma equals:', gam)
    for train_sector, test_sector in kf.split(ds):
        ds_train, ds_test = ds[train_sector], ds[test_sector]
        dl_train, dl_test = dl[train_sector], dl[test_sector]
        svmcfr = svmm.fit(ds_train,dl_train)
        svm_result = svmcfr.predict(ds_test)
        meansv.append(accuracy_score(dl_test, svm_result))
        print ('accuracy for fold ' ,fold, ' : {}'.format(accuracy_score(dl_test, svm_result)))
        fold += 1
    print('Mean accuracy for Gamma equals ', gam, ' in 5-fold cross validation:', sum(meansv)/len(meansv))  

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
for gama in [1,0.1,0.01,0.001]:
    CVSVM(gama, data, stress)
print('There is no difference can be observed when we use different Gamma ')

#fitting the data
print('Fitting the dataset')
svmclf = svm.SVC(gamma=0.001,kernel='rbf')
svmclf.fit(data, stress)

#training dataset accuracy
print('For training dataset')
svmtrain_result = svmclf.predict(data)
print ('training dataset accuracy: {}'.format(accuracy_score(stress, svmtrain_result)))

#confusion matrix + classification report
print("Classification report for classifier %s:\n%s\n"
      % (svmclf, metrics.classification_report(stress, svmtrain_result)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(stress, svmtrain_result))

#testing dataset accuracy
print('For testing dataset')
svsvm_result = svmclf.predict(dota)
print ('testing dataset accuracy: {}'.format(accuracy_score(tension, svsvm_result)))

#confusion matrix + classification report
print("Classification report for classifier %s:\n%s\n"
      % (svmclf, metrics.classification_report(tension, svsvm_result)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(tension, svsvm_result))

e1 = timeit.default_timer()
print ('Training time: {}'.format(e1-start))