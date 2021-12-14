
# coding: utf-8

# In[6]:


import pandas as pd
#train
trainenroll = pd.read_csv('enrollment_train.csv')
trainlog = pd.read_csv('log_train.csv')
traintruth =  pd.read_csv('truth_train.csv')
#test
testenroll = pd.read_csv('enrollment_test.csv')
testlog = pd.read_csv('log_test.csv')
testtruth =  pd.read_csv('truth_test.csv')
#coursedate
coursedate= pd.read_csv('date.csv')

