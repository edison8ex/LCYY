rm(list=ls(all=TRUE))
par(mar=c(2,2,2,2),mfrow=c(1,1))  #set margin for plotting

#1.1 Load data
heart <- read.csv('RMBI3000B/Project/Data/heart.csv')
heart
#age = age, sex: 1=male, 0=female, cp=chest pain type, testbps=resting blood pressure, serum cholestoral, 
#fbs = fasting blood suger > 120: 1=True, 0=False, restecg=resting electrocardiographic results, 
#thalach = maximum heart rate achieved, exang = execrise induced angina(1=Yes,0=No), 
#oldpeak= ST depression induced by execrise ST segment, ca= number of major vessels (0-3) colored by flourosopy,
#thal = 3 = normal; 6 = fixed defect; 7 = reversable defect, target: 1-have heart diease, 0 does not

#1.2 use barplot to see the distribution of targeted variable
tartable = table(heart$target)
barplot(tartable, main="Distribution of people who have heart diease and not")

#1.3 
#install package ROSE to perform oversampling
#install.packages("ROSE") #Randomly Over Sampling Example
library(ROSE)
b_df <- ovun.sample(target~., data = heart, method = "over", N=2*max(tartable))$data
baltable = table(b_df$target)
barplot(baltable, main="Distribution of target variable in balanced dataset")

#1.4
str(b_df)
#Change target variable to catagorical for fitting in model and prediction
b_df$target = as.factor(b_df$target)
str(b_df)


#2.1
#Spliting dataset into 70% train, 30% test
sample_size <- floor(0.7*nrow(b_df))
set.seed(101) #make the split be reproducible (split result will be the same every time)
train_index <- sample(seq_len(nrow(b_df)),size = sample_size)
train = b_df[train_index,]
test = b_df[-train_index,]

#2.2
#Fit the model and Perform prediction
#install.packages("e1071")
library(e1071)
NB_model = naiveBayes(target~. , data=train)
#result on training set
NB_model
NB_predict = predict(NB_model,train)
#confusion matrix
table(NB_predict, train$target)
#result on testing set
NB_model
NB_predict = predict(NB_model,test)
#confusion matrix
confu_mtx = table(NB_predict, test$target)
confu_mtx

#2.3
#calculate accuracy, recall, precision and f1 score
TP = confu_mtx[[1,1]]; TN = confu_mtx[[2,2]];
FP = confu_mtx[[1,2]]; FN = confu_mtx[[2,1]];
N = sum(confu_mtx)
accuracy = (TP+TN)/N; precision = TP/(TP+FP);
recall = TP/(TP+FN);
f1 = (2*precision*recall)/(precision+recall)
cat("Accuracy:",accuracy,"Precision:",precision,"Recall:", recall, "F1-score:", f1)
