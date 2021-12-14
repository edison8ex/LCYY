
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import date
from datetime import datetime
from imblearn.over_sampling import SMOTE
get_ipython().run_line_magic('run', 'read_data.ipynb')


# In[2]:


trainenroll


# In[3]:


trainlog


# In[4]:


traintruth


# In[5]:


testenroll


# In[6]:


testlog


# In[7]:


testtruth


# In[8]:


#enrollment id + courseid
trainenroll.index = trainenroll['enrollment_id']
courseiddf = trainenroll.drop(['enrollment_id','username'],axis=1)
#course id
gen_dummy_features = pd.get_dummies(courseiddf['course_id'], drop_first=False)
#num event + num source
dumlist = ['event','source']
tempdum = pd.get_dummies(trainlog[dumlist], drop_first=False)
daa = pd.concat([trainlog['enrollment_id'],tempdum], axis=1)
#print(daa)
dfcc = daa.groupby('enrollment_id').sum()
#standardization
xx = pd.DataFrame()
num_feat = ['event_access','event_discussion','event_navigate','event_page_close','event_problem','event_wiki','source_browser','source_server']
for eachc in num_feat:
    db = []
    for eachr in dfcc[eachc]:
        db.append((eachr-dfcc[eachc].min())/(dfcc[eachc].max()-dfcc[eachc].min()))
    xx[eachc] = db
xx.index = trainenroll['enrollment_id']
#concat df
datest = pd.concat([gen_dummy_features,xx], axis=1)
#variable to check if a student did not use the MOOC for 10 consecutive days
drop=[]
enroll = trainenroll.reset_index(drop=True)
for i in range(enroll.shape[0]):
    course_id=enroll.course_id[i]
    enrollment_id=enroll.enrollment_id[i]
    ending=datetime.strptime(coursedate[coursedate.course_id==course_id]['to'].to_string(index=False),"%Y-%m-%d").date()
    drop_by_day=0
    df=trainlog[trainlog.enrollment_id==enrollment_id]
    d0=datetime.strptime(df.time.iloc[df.shape[0]-1], "%Y-%m-%dT%H:%M:%S").date()
    delta=ending-d0
    if delta.days>10: drop_by_day=1
    for j in range(df.shape[0]):
        if j+1<df.shape[0]:
            d0=datetime.strptime(df.time.iloc[j], "%Y-%m-%dT%H:%M:%S").date()
            d1=datetime.strptime(df.time.iloc[j+1], "%Y-%m-%dT%H:%M:%S").date()
            delta=d1-d0
        if delta.days>10: drop_by_day=1
    drop.append([drop_by_day])
drop_df=pd.DataFrame(data=drop, columns=['drop'])
drop_df.index = enroll.enrollment_id
datest['skip10']=drop_df
X_tr = datest
X_train = X_tr.reset_index()
X_train


# In[9]:


#amendment on truth
traintruth = traintruth.rename(index=str, columns={'1': "enrollment_id", "0": "Drop"})
ssa = pd.DataFrame([[1, 0]], columns=['enrollment_id','Drop'])
ddf3 = traintruth.append(ssa)
ddf3.reset_index(drop=True)
y = ddf3.sort_values(by=['enrollment_id'])
y = y.reset_index(drop=True)
y_train = y['Drop']
y_train
#y


# In[10]:


#enrollment id + courseid
testenroll.index = testenroll['enrollment_id']
tcourseiddf = testenroll.drop(['enrollment_id','username'],axis=1)
#course id
t_gen_dummy_features = pd.get_dummies(tcourseiddf['course_id'], drop_first=False)
#num event + num source
dumlist = ['event','source']
ttempdum = pd.get_dummies(testlog[dumlist], drop_first=False)
tdaa = pd.concat([testlog['enrollment_id'],ttempdum], axis=1)
#print(daa)
tdfcc = tdaa.groupby('enrollment_id').sum()
#standardization
txx = pd.DataFrame()
num_feat = ['event_access','event_discussion','event_navigate','event_page_close','event_problem','event_wiki','source_browser','source_server']
for eachc in num_feat:
    db = []
    for eachr in tdfcc[eachc]:
        db.append((eachr-tdfcc[eachc].min())/(tdfcc[eachc].max()-tdfcc[eachc].min()))
    txx[eachc] = db
txx.index = testenroll['enrollment_id']
#concat df
tdatest = pd.concat([t_gen_dummy_features,txx], axis=1)
#variable to check if a student did not use the MOOC for 10 consecutive days
drop=[]
enroll = testenroll.reset_index(drop=True)
for i in range(enroll.shape[0]):
    course_id=enroll.course_id[i]
    enrollment_id=enroll.enrollment_id[i]
    ending=datetime.strptime(coursedate[coursedate.course_id==course_id]['to'].to_string(index=False),"%Y-%m-%d").date()
    drop_by_day=0
    df=testlog[testlog.enrollment_id==enrollment_id]
    d0=datetime.strptime(df.time.iloc[df.shape[0]-1], "%Y-%m-%dT%H:%M:%S").date()
    delta=ending-d0
    if delta.days>10: drop_by_day=1
    for j in range(df.shape[0]):
        if j+1<df.shape[0]:
            d0=datetime.strptime(df.time.iloc[j], "%Y-%m-%dT%H:%M:%S").date()
            d1=datetime.strptime(df.time.iloc[j+1], "%Y-%m-%dT%H:%M:%S").date()
            delta=d1-d0
        if delta.days>10: drop_by_day=1
    drop.append([drop_by_day])
t_drop_df=pd.DataFrame(data=drop, columns=['drop'])
t_drop_df.index = enroll.enrollment_id
tdatest['skip10']=t_drop_df
X_t = tdatest
X_test = X_t.reset_index()
X_test


# In[11]:


#amendment on truth
testtruth = testtruth.rename(index=str, columns={'30': "enrollment_id", "0": "Drop"})
tssa = pd.DataFrame([[30, 0]], columns=['enrollment_id','Drop'])
tddf3 = testtruth.append(tssa)
tddf3.reset_index(drop=True)
y = tddf3.sort_values(by=['enrollment_id'])
y = y.reset_index(drop=True)
y_test = y['Drop']
y_test


# In[14]:


#SMOTE
smote = SMOTE(ratio='minority')
X_sm, y_sm = smote.fit_sample(X_train, y_train)
Xos = pd.DataFrame(X_sm,columns=X_train.columns)
catelist = ['1pvLqtotBsKv7QSOsLicJDQMHx3lui6d',
       '3VkHkmOtom3jM2wCu94xgzzu1d6Dn7or', '3cnZpv6ReApmCaZyaQwi2izDZxVRdC01',
       '5Gyp41oLVo7Gg7vF4vpmggWP5MU70QO6', '5X6FeZozNMgE2VRi3MJYjkkFK8SETtu2',
       '7GRhBDsirIGkRZBtSMEzNTyDr2JQm4xx', '81UZtt1JJwBFYMj5u38WNKCSVA4IJSDv',
       '9Bd26pfDLvkPINwLnpaGcf0LrLUvY1Mz', '9Mq1P5hrrLw6Bh9X4W4ZjisQJDdxjz9x',
       '9zpXzW9zCfU8KGBWkhlsGH8B8czISH4J', 'A3fsA9Zfv1X2fVEQhTw51lKENdNrEqT3',
       'AXUJZGmZ0xaYSWazu8RQ1G5c76ECT1Kd', 'DABrJ6O4AotFwuAbfo1fuMj40VmMpPGX',
       'DPnLzkJJqOOPRJfBxIHbQEERiYHu5ila', 'Er0RFawC4sHagDmmQZcBGBrzamLQcblZ',
       'G8EPVSXsOYB5YQWZGiz1aVq5Pgr2GrQu', 'H2lDW05SyKnwntZ6Fora76aPAEswcMa5',
       'HbeAZjZFFQUe90oTP0RRO0PEtRAqU3kK', 'I7Go4XwWgpjRJM8EZGEnBpkfSmBNOlsO',
       'KHPw0gmg1Ad3V07TqRpyBzA8mRjj7mkt', 'NmbZ3BmS8V4pMg6oxXHWpqqMZCE1jvYt',
       'RXDvfPUBYFlVdlueBFbLW0mhhAyGEqpt', 'SpATywNh6bZuzm8s1ceuBUnMUAeoAHHw',
       'TAYxxh39I2LZnftBpL0LfF2NxzrCKpkx', 'V4tXq15GxHo2gaMpaJLZ3IGEkP949IbE',
       'WM572q68zD5VW8pcvVTc1RhhFUq3iRFN', 'Wm3dddHSynJ76EJV6hyLYKGGRL0JF3YK',
       'X78EhlW2JxwO1I6S3U4yZVwkEQpKXLOj', 'a2E7NQC7nZB7WHEhKGhKnKvUWtsLAQzh',
       'bWdj2GDclj5ofokWjzoa5jAwMkxCykd6', 'fbPkOYLVPtPgIt0MxizjfFJov3JbHyAi',
       'gvEwgd64UX4t3K7ftZwXiMkFuxFUAqQE', 'mTmmr5zd8l4wXhwiULwjSmSbi9ktcFmV',
       'nSfGxfEtzw5G72fVbfaowxsV46Pg1xIc', 'q6A6QG7qMpyNcznyT2XaIxnfNGkZRxXl',
       'shM3Yy9vxHn2aqjSYfQXOcwGo0hWh3MI', 'tXbz2ZYaRyb2ZsWUBPoYzAmisOhHQrYl',
       'xMd9DzNyUCTLRPVbwWVzf4vq06oqrTT1', 'ykoe1cCWK134BJmfbNoPEenJOIWdtQOZ',
           'skip10']
for e in catelist:
    Xos = Xos.round({e:0})
X_trainos = Xos.drop(['enrollment_id'],1)
X_testos = X_test.drop(['enrollment_id'],1)


# In[ ]:


y_trainos=pd.DataFrame(y_sm, columns=["Drop"])


# In[24]:


X_train.to_csv('X_train.txt', sep=',',index=False)
X_test.to_csv('X_test.txt', sep=',',index=False)
y_train.to_csv('y_train.txt', sep=',',index=False,header='Drop')
y_test.to_csv('y_test.txt', sep=',',index=False,header='Drop')
X_trainos.to_csv('X_trainos.txt', sep=',',index=False)
X_testos.to_csv('X_testos.txt', sep=',',index=False)
y_trainos.to_csv('y_trainos.txt', sep=',',index=False,header='Drop')

