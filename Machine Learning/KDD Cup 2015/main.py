import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from scipy import stats
import statsmodels.api as sm
from sklearn import metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

def vifcl(df):
    a = 0
    i = 0
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    vif["features"] = df.columns
    t = vif["VIF Factor"].max()
    for each in vif["VIF Factor"]:
        if each == t and each >= 10:
            a = i
        i += 1
    vax = vif["VIF Factor"].loc[a]
    print(vif.loc[a],"\n>10")
    return a,vax

X_test = pd.read_csv('X_test.txt')
X_train = pd.read_csv('X_train.txt')
y_test = pd.read_csv('y_test.txt')
y_train = pd.read_csv('y_train.txt')
X_trainos = pd.read_csv('X_trainos.txt')
X_testos = pd.read_csv('X_testos.txt')
y_testos = pd.read_csv('y_testos.txt')
y_trainos = pd.read_csv('y_trainos.txt')

#Logistic regression
X_trainlog = X_train.drop(['enrollment_id'],axis=1)
y_trainlog = np.array(y_train)
X_testlog = X_test.drop(['enrollment_id'],axis=1)
y_testlog = np.array(y_test)
X_trainlog = X_trainos
y_trainlog = np.array(y_trainos)
X_testlog = X_testos
y_testlog = np.array(y_testos)
dropcindex=[]
while True:
    vmax = 0
    dropt = False
    k,s = vifcl(X_trainlog)
    print(s)
    if s >= 10:
        dropt = True
    if dropt == True:
        print("dropped",X_trainlog.columns.get_values()[k])
        X_trainlog = X_trainlog.drop([X_trainlog.columns.get_values()[k]],axis=1)
        vif = vif.drop([k])
        X_trainlog = X_trainlog.reset_index(drop=True)
        vif = vif.reset_index(drop=True)
        dropcindex.append(k)
    else:
        break
sm_model = sm.Logit(y_trainlog, sm.add_constant(X_trainlog)).fit(disp=0)
sm_model.summary()
# For each Xi, calculate VIF
vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_trainlog.values, i) for i in range(X_trainlog.shape[1])]
vif["features"] = X_trainlog.columns
vif.round(1)
sm_model = sm.Logit(y_trainlog, sm.add_constant(X_trainlog)).fit(disp=0)
sm_model.summary()
#fit the cleansed data again
model = LogisticRegression(C = 1000000000,tol=0.001)
model = model.fit (X_trainlog,y_trainlog.ravel())
probability = model.predict_proba(X_trainlog)
predicted = model.predict(X_trainlog)
## Evaluate The Model Confusion Matrix
print (metrics.confusion_matrix(y_trainlog, predicted))
## Classification Report
print (metrics.classification_report(y_trainlog, predicted))
## Model Accuracy
print ('Accuracy:',model.score(X_trainlog,y_trainlog))
## Prediction Performance – ROC curve & AUC
y_predict_probabilities = probability[:,1]
fpr, tpr, _ = roc_curve(y_trainlog, y_predict_probabilities)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
X_testlog = X_testlog.drop(['source_server','source_browser'],axis=1)
tprobability = model.predict_proba(X_testlog)
tpredicted = model.predict(X_testlog)
## Evaluate The Model Confusion Matrix
print (metrics.confusion_matrix(y_testlog, tpredicted))
## Classification Report
print (metrics.classification_report(y_testlog, tpredicted))
## Model Accuracy
print ('Accuracy:',model.score(X_testlog,y_testlog))
## Prediction Performance – ROC curve & AUC
y_predict_probabilities = tprobability[:,1]
fpr, tpr, _ = roc_curve(y_testlog, y_predict_probabilities)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
#Neural Network
y_trainnn = np.array(y_trainos)
y_testnn = np.array(y_testos)
X_trainnn = X_trainos
X_testnn = X_testos
nnclf = MLPClassifier(hidden_layer_sizes=(200,200,25), random_state=1,alpha=1e-6,max_iter=400)
nnclf.fit(X_trainnn, y_trainnn.ravel())
#training dataset classfication report
print('For training dataset')
nntrain_result = nnclf.predict(X_trainnn)
print ('training dataset accuracy: {}'.format(accuracy_score(y_trainnn, nntrain_result)))
print('\n')
print("Classification report for classifier %s:\n%s\n"
      % (nnclf, metrics.classification_report(y_trainnn, nntrain_result)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_trainnn, nntrain_result))
#testing dataset classfication report
print('For testing dataset')
tnntrain_result = nnclf.predict(X_testnn)
print ('testing dataset accuracy: {}'.format(accuracy_score(y_testnn, tnntrain_result)))
print('\n')
print("Classification report for classifier %s:\n%s\n"
      % (nnclf, metrics.classification_report(y_testnn, tnntrain_result)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_testnn, tnntrain_result))
#Random Forest
y_trainrf = np.array(y_train)
y_testrf = np.array(y_test)
X_trainrf = X_train
X_testrf = X_test
clf = RandomForestClassifier(random_state=0, max_features = None,class_weight="balanced", n_estimators=80, max_depth=63, min_samples_leaf=2,min_samples_split=5, bootstrap=True,criterion="entropy")
RF = clf.fit(X_trainrf, y_trainrf)
#test dataset
print('For test dataset')
train_result = RF.predict(X_testrf)
print ('training dataset accuracy: {}'.format(accuracy_score(y_testrf, train_result)))
print('\n')
print("Classification report for classifier %s:\n%s\n"
      % (RF, metrics.classification_report(y_testrf, train_result)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_testrf, train_result))
probability = clf.predict_proba(X_testrf)
predicted = clf.predict(X_testrf)
y_predict_probabilities = probability[:,1]
fpr, tpr, _ = roc_curve(y_testrf, y_predict_probabilities)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
