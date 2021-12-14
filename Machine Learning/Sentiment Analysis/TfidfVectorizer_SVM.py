from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.externals import joblib

review_train=pd.read_csv("data/train.csv")
review_test=pd.read_csv("data/valid.csv")

x1 = review_train["text"]
y = review_train["stars"]
vect = TfidfVectorizer(ngram_range=(1,3),max_df=0.25)
x_vect1 = vect.fit_transform(x1)
x1_test = review_test["text"]
y_test = review_test["stars"]
x_vect1_test = vect.transform(x1_test)

classf = LinearSVC(class_weight="balanced")
classf.fit(x_vect1, y)
pred = classf.predict(x_vect1_test)
print("Linear SVC:",accuracy_score(y_test, pred))

joblib.dump( classf,'linearSVC_maxdf0.25.pkl')
