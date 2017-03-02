---
title: "Untitled"
author: "Marit Sandstad"
date: "February 22, 2017"
output: html_document
---



# Correct exercise performance prediction (Human activity recognition)

This report is considers a data set of human exercise monitoring from correctly and incorrectly performed exercises to find a machine learning prediction model which can classify exercises into correct or one of four different categories for incorrect performance of the same exercise. After some preprocessing we find that a random forest model fitted 95% principal component set of the numeric measurements gives a very good classification fit (96% predicted test accuracy). We also find that a support vector machine algorithm performs quite well, with much better computational speed. A general boost model and a linear discriminant analysis model are also performed, both with significantly worse performance. Finally we have also fitted a stacked model which performs on par with the random forest model, however as this is less computationally efficient still, we choose the random forest model as our final prediction model. 

## Note on the data set

We use the data kindly provided by [Groupware\@LES](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv). See also:

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. **Qualitative Activity Recognition of Weight Lifting Exercises.** Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.

[Read more:](http://groupware.les.inf.puc-rio.br/har#weight_lifting_exercises#ixzz4ZPMlGZEc)


## Loading test and training sets


```r
library(caret)
tmp <- tempfile()
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", tmp)
trainSet <- read.csv(tmp, stringsAsFactors = FALSE, na.strings = c("", "#DIV/0!"))
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", tmp)
testSet <- read.csv(tmp, stringsAsFactors = FALSE, na.strings = c("", "#DIV/0!"))
unlink(tmp)
trainSet$classe <- factor(trainSet$classe)
```

## Preprocessing

### Taking out na and character variables
In order for the data to be more workable. A quick look at the data reveals that many colomns are mostly empty or cotain almost only NA. 

```r
head(trainSet)
```

I will remove all these variables that include missing data. 

In addition the first 7 columns contain username, data set numbering and time-stamp data which seem uncorrelated with the classification. I will remove these as well. 

For reasons of clarification I will rename the test set the prob or problem set (as I will need a bigger test set to test my actual model on).

```r
naInd <- which(apply(is.na(trainSet), 2, sum)>0)
chrInd <- which(sapply(trainSet, is.character) > 0)
trainCut <- trainSet[, -c(1:7, naInd, chrInd)]
probCut <- testSet[, -c(1:7, naInd, chrInd)]
lenCut <- dim(trainCut)[2]
```

### Partitioning the training set and other preprocessing
I partition the training data set into 60 % training, 20 % validation and 20 % test components. However, our data set still has a very large number of variables, and as I have no expert knowledge to guide me, I start by performing principal component analysis on the new training set to get a more minimal variable set. Then I apply this transformation to the other partitions:

```r
set.seed(25535)
inTrain <- createDataPartition(y = trainCut$classe, p = 0.6, list = FALSE)
training <- trainCut[inTrain, ]
notTraining <- trainCut[-inTrain, ]
inVal <- createDataPartition(y = notTraining$classe, p = 0.5, list = FALSE)
validation <- notTraining[inVal, ]
testing <- notTraining[-inVal, ]

preProc <- preProcess(training[, -lenCut], method = c("center", "scale", "pca"))
trainPca <- predict(preProc, newdata = training[, -lenCut])
trainPca$classe <- training$classe
valPca <- predict(preProc, newdata = validation[, -lenCut])
valPca$classe <- validation$classe
testPca <- predict(preProc, newdata = testing[, -lenCut])
testPca$classe <- testing$classe
probPca <- predict(preProc, probCut[,-lenCut])
probPca$problem_id <- probCut$problem_id
```

## Model fitting
We fit four different machine learning models svm, lda, rf and gbm, and finally a stacked model from the predictions from each of these on the training set. We compare their performance on the validation set to choose a model.

We start by fitting an svm model to the data and find its accuracy for training and validation data:

```r
library(e1071)
fitsvm <- svm(classe ~., data = trainPca)
predsvm <- predict(fitsvm)
predValsvm <- predict(fitsvm, newdata = valPca)
confusionMatrix(predsvm, trainPca$classe)$overall["Accuracy"]
```

```
##  Accuracy 
## 0.9457371
```

```r
confusionMatrix(predValsvm, valPca$classe)$overall["Accuracy"]
```

```
##  Accuracy 
## 0.9390772
```

Then I fit a linear discriminant analysis model and find its accuracy on training and validation data:

```r
library(MASS)
fitlda <- train(classe ~., data = trainPca, method = "lda")
predlda <- predict(fitlda)
predVallda <- predict(fitlda, newdata = valPca)
confusionMatrix(predlda, trainPca$classe)$overall["Accuracy"]
```

```
##  Accuracy 
## 0.5293818
```

```r
confusionMatrix(predVallda, valPca$classe)$overall["Accuracy"]
```

```
##  Accuracy 
## 0.5409126
```

Then I fit a random forest model and find its accuracy for training and validation data:

```r
library(randomForest)
fitRF <- train(classe~. , method = "rf", data = trainPca)
predRF <- predict(fitRF)
predValRF <- predict(fitRF, newdata = valPca)
confusionMatrix(predRF, trainPca$classe)$overall["Accuracy"]
```

```
## Accuracy 
##        1
```

```r
confusionMatrix(predValRF, valPca$classe)$overall["Accuracy"]
```

```
##  Accuracy 
## 0.9717053
```

Then we fit a generalized boost model and find its accuracy for training and validation data:

```r
library(gbm)
fitGBM <- train(classe~. , method = "gbm", data = trainPca)
predgbm <- predict(fitGBM)
predValgbm <- predict(fitGBM, newdata = valPca)
```


```r
confusionMatrix(predgbm, trainPca$classe)$overall["Accuracy"]
```

```
## Accuracy 
## 0.856233
```

```r
confusionMatrix(predValgbm, valPca$classe)$overall["Accuracy"]
```

```
##  Accuracy 
## 0.8236044
```

Finally we also fit a stacked model using random forest with the predictions from all these three models:

```r
stackTraining<- data.frame(predsvm = predsvm, predlda = predlda, predgbm = predgbm, predRF = predRF, classe = trainPca$classe)

fitStack <- train(classe~., method = "rf", data = stackTraining)
confusionMatrix(predict(fitStack), trainPca$classe)$overall["Accuracy"]
```

```
## Accuracy 
##        1
```

```r
stackVal <- data.frame(predsvm = predValsvm, predlda = predVallda, predgbm = predValgbm, predRF = predValRF, classe = valPca$classe)

confusionMatrix(predict(fitStack, newdata = stackVal), valPca$classe)$overall["Accuracy"]
```

```
##  Accuracy 
## 0.9717053
```

## Conclusion
I find that the first random forest model performs the best on both test and validation data. The stacked model performs equally well, but as it is more computationally demanding, we choose the simpler model instead. Hence this the random forest is the one we want to use for our final predictions. (Note that the svm model also performs quite well and is much faster, so if speed is an issue, the svm model would also be a good choice). To assess the test accuracy of the final model we test it on our test set:


```r
confusionMatrix(predict(fitRF, newdata = testPca), testPca$classe)$overall["Accuracy"]
```

```
##  Accuracy 
## 0.9681366
```

