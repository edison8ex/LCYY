rm(list=ls(all=TRUE))
rawdf <- read.csv('RMBI3000B/Project/Data/Youtube_data2.csv')
rawdf

#Preprocessing, change data type
rawdf$Video.Uploads = as.numeric(rawdf$Video.Uploads)
rawdf$Subscribers = as.numeric(rawdf$Subscribers)
rawdf$Video.views = as.numeric(rawdf$Video.views)
str(rawdf)

#Exploratory Data Analysis
par(mar=c(2,2,2,2),mfrow=c(1,1))  #set margin for plotting
barplot(rawdf$Video.views, main="Vidoe Views of different Youtubers") #remove extreme data

#By performing a boxplot on target variable, we can see the dataset is not a balanced dataset
barplot(table(rawdf$Grade), main="Grades Youtube assigned to different Youtubers")

#Data Cleansing
#drop extreme value in Video.view
p_df = rawdf[-which.max(rawdf$Video.views),] #- operator exclude index
#drop blank level in target variable
gradevec = c("A++ ", "A+ ", "A ", "A- ", "B+ ")
p_df = p_df[which(p_df$Grade %in% gradevec),]
#drop blank numerical data in Subscribers
p_df = p_df[-which(p_df$Subscribers == 1),]

#Comparsion
table(rawdf$Grade) #before
table(p_df$Grade) #after

#After cleasing, extreme and noise data removed
barplot(p_df$Video.views, main="Vidoe Views of different Youtubers")
barplot(table(p_df$Grade), main="Grades Youtube assigned to different Youtubers")

#Drop unnessary columns
c_df = p_df[,c(-1)]

#Oversampling
#install.packages("UBL")

library(UBL)
#Using the Synthetic Minority Over-sampling Technique to perform oversampling
newdd <- SmoteClassif(Grade~., c_df, "balance",k=7)
barplot(table(newdd$Grade), main="Grades Youtube assigned to different Youtubers")


#Spliting dataset into 70% train, 30% test
sample_size <- floor(0.7*nrow(newdd))
set.seed(101) #make the split be reproducible (split result will be the same every time)
train_index <- sample(seq_len(nrow(newdd)),size = sample_size)
train = newdd[train_index,]
test = newdd[-train_index,]

#install.packages("e1071")
library(e1071)
NB_model = naiveBayes(Grade~. , data=train)
#result on training set
NB_model
NB_predict = predict(NB_model, train)

#confusion matrix
table(NB_predict, train$Grade)[-1,-1]

#result on testing set
NB_predict_test = predict(NB_model, test)

#confusion mtx
conf_mtx = table(NB_predict_test, test$Grade)[-1,-1]
conf_mtx

#Accuracy
t = 0
for (k in seq(nrow(conf_mtx))) {
  t = t + conf_mtx[[k,k]]
}
N = sum(conf_mtx)
acc = t/N

#recall
rclist = c()
for (i in seq(nrow(conf_mtx))) {
  rsum = 0
  tt = conf_mtx[[i,i]]
  for (j in seq(ncol(conf_mtx))) {
    rsum = rsum + conf_mtx[[i,j]]
  }
  rc = tt/rsum
  rclist = append(rclist, rc)
}
recall = mean(rclist)

#precision
plist = c()
for (j in seq(ncol(conf_mtx))) {
  csum = 0
  tt = conf_mtx[[j,j]]
  for (i in seq(nrow(conf_mtx))) {
    csum = csum + conf_mtx[[i,j]]
  }
  pc = tt/csum
  plist = append(plist, pc)
}
precision = mean(plist)

cat("Accuracy:", acc, "Recall:", recall, "Precision:", precision)

