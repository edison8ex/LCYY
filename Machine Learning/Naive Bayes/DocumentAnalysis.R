rm(list=ls(all=TRUE))
library(doMC)
registerDoMC(cores=detectCores()-2)
movie<- read.csv("RMBI3000B/Project/Data/movie-pang02.csv", stringsAsFactors = FALSE)
str(movie)

#Randomize dataset
set.seed(111)
movie <- movie[sample(nrow(movie)), ]
str(movie)

#Convert the target variable into factor
movie$class <- as.factor(movie$class)

#install.packages("tm")
library(tm)
library(magrittr)
corpus <- Corpus(VectorSource(movie$text)) #structure for managing text documents, like tensor in tensorflow
#Data Cleansing, remove numbers, punctuation, white space, converting to lower case and discard common stopwords
corpus.clean <- corpus %>%
  tm_map(content_transformer(tolower)) %>% 
  tm_map(removePunctuation) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stripWhitespace)

#Embedding
DocToMtx <- DocumentTermMatrix(corpus.clean)

#Split data into train and test
sample_size <- floor(0.7*nrow(movie))
train_index <- sample(seq_len(nrow(movie)),size = sample_size)
movie.train <- movie[train_index,]
movie.test <- movie[-train_index,]
DocToMtx.train = DocToMtx[train_index,]
DocToMtx.test = DocToMtx[-train_index,]
corpus.clean.train <- corpus.clean[train_index]
corpus.clean.test <- corpus.clean[-train_index]

#Feature selection
dim(DocToMtx.train)
#The embedding contains 38957 features but not all of them will be useful for classification. 
#We can reduce the number of features by ignoring words which appear in less than a certain threshold.
#Lets use threshold = 5
fivefreq <- findFreqTerms(DocToMtx.train, 5)
length((fivefreq))
#Only 11706 features/words appear more than 5 times, they will be more significant features in building model
#Create the embedding for training dataset using these features
DocToMtx.train.nb <- DocumentTermMatrix(corpus.clean.train, control=list(dictionary = fivefreq))
dim(DocToMtx.train.nb)
#Create the embedding for testing dataset using these features
DocToMtx.test.nb <- DocumentTermMatrix(corpus.clean.test, control=list(dictionary = fivefreq))
dim(DocToMtx.test.nb)

#Matching words in df from embedding layer
# Function to convert the word frequencies to yes (presence) and no (absence) labels
convert_count <- function(x) {
  y <- ifelse(x > 0, 1,0)
  y <- factor(y, levels=c(0,1), labels=c("No", "Yes"))
  y
}
trainNB <- apply(DocToMtx.train.nb, 2, convert_count)
testNB <- apply(DocToMtx.test.nb, 2, convert_count)

#Fitting the model and performing prediction
library(e1071)
classifier <- naiveBayes(trainNB, movie.train$class, laplace = 1)
pred <- predict(classifier, newdata=testNB)

#Results
#install.packages("caret")
library(caret)
conf.mtx <- confusionMatrix(pred, movie.test$class)
conf.mtx
