---
title: "PracticalMachineLearning"
author: "JFlanagan"
date: "Saturday, October 25, 2014"
output: html_document
---

##Practical Machine Learning from www.coursera.org
###Course Project

The objective of this project is to develop a predictive model to predict the quality of barbell lifts using  data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants, who were requested to perform barbell lifts correctly and incorrectly in 5 different ways. 

The steps undertaken in this project were:

1. Import training data
2. Tidy data (removal of NAs and non-predictive information)
3. Split data 70/30 for the purposes of model building and cross-validation
4. Apply different predictive models and assess accuracy based on "Out of Sample Error"
5. Import test data and tidy in the same manner as the training data
6. Apply predictive model which gave best results in step 4.

###Step 1. Import training data

```r
trainingraw <- read.csv("C:/Users/mikoflan/Documents/PredMachLearning/pml-training.csv", 
                          na.strings= c("NA",""," "))
```
###Step 2. Tidy data (removal of NAs and non-predictive information)
First, I computed the number of missing values per column of data. Then I investigated the occurence of NAs; the results showed that columns either contained >15,000 NAs or no NAs. In a dataset with 19,622 observations, I decided that if there were in excess of 15,000 NAs, these columns added little predictive power to the model chosen.


```r
dim(trainingraw)
```

```
## [1] 19622   160
```

```r
trainingrawNA <- apply(trainingraw, 2, function(x) sum(is.na(x)))

sum(trainingrawNA > 10)
```

```
## [1] 100
```

```r
sum(trainingrawNA > 15000)
```

```
## [1] 100
```

```r
sum(trainingrawNA > 0) ## therefore can remove all cols with any NA (if one NA, then > 10000 NAs)
```

```
## [1] 100
```

```r
training <- trainingraw[,which(trainingrawNA < 1)]

ncoltraining <- ncol(training)
training <- training[,8:ncoltraining] ##remove non-predictive information in 1st 8 columns
table(training$classe) ## get overview of outcome
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```
###Step 3. Split data 70/30 for the purposes of model building and cross-validation

```r
library(caret)
inTrain <- createDataPartition(y = training$classe, p = 0.7, list = FALSE)
train <- training[inTrain, ]
crossval <- training[-inTrain, ]

library(corrplot)
train2 <- train[,-53]
M <- cor(train2)
corrplot(M, type="lower", order="hclust") ## visualize potential correlations
```

![plot of chunk unnamed-chunk-3](figure/unnamed-chunk-3.png) 

###Step 4. Apply different predictive models and assess accuracy based on "Out of Sample Error"

I would like to Out of Sample Error rate of less than 5 %.

4a. Linear Discriminant Analysis

```r
modlda <- train(classe ~ ., data=train, method = "lda")
plda <- predict(modlda, crossval)
confusionMatrix(crossval$classe, plda) ## 70.28% accuracy
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1361   30  134  138   11
##          B  169  747  136   40   47
##          C  104  117  660  118   27
##          D   56   35  123  705   45
##          E   39  194  109  103  637
## 
## Overall Statistics
##                                        
##                Accuracy : 0.698        
##                  95% CI : (0.686, 0.71)
##     No Information Rate : 0.294        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.618        
##  Mcnemar's Test P-Value : <2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.787    0.665    0.568    0.639    0.831
## Specificity             0.925    0.918    0.923    0.946    0.913
## Pos Pred Value          0.813    0.656    0.643    0.731    0.589
## Neg Pred Value          0.913    0.921    0.897    0.919    0.973
## Prevalence              0.294    0.191    0.197    0.188    0.130
## Detection Rate          0.231    0.127    0.112    0.120    0.108
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.856    0.791    0.745    0.792    0.872
```

```r
OOSE.acc.lda <- sum(plda == crossval$classe)/length(plda)
OOSE.lda <- round(((1-OOSE.acc.lda)*100), digits=2)
```
The Out of Sample Error using Linear Discriminant Analysis was estimated to be 30.16%.

4b. Classification tree

```r
library(rpart)
modrpart <- train(classe ~., method="rpart", data=train)
predrpart <- predict(modrpart, crossval)
confusionMatrix(crossval$classe, predrpart) ##49% accuracy
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1503   31  134    0    6
##          B  463  310  366    0    0
##          C  488   18  520    0    0
##          D  462  152  350    0    0
##          E  158   73  384    0  467
## 
## Overall Statistics
##                                         
##                Accuracy : 0.476         
##                  95% CI : (0.463, 0.489)
##     No Information Rate : 0.522         
##     P-Value [Acc > NIR] : 1             
##                                         
##                   Kappa : 0.315         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.489   0.5308   0.2965       NA   0.9873
## Specificity             0.939   0.8436   0.8775    0.836   0.8864
## Pos Pred Value          0.898   0.2722   0.5068       NA   0.4316
## Neg Pred Value          0.627   0.9423   0.7460       NA   0.9988
## Prevalence              0.522   0.0992   0.2980    0.000   0.0804
## Detection Rate          0.255   0.0527   0.0884    0.000   0.0794
## Detection Prevalence    0.284   0.1935   0.1743    0.164   0.1839
## Balanced Accuracy       0.714   0.6872   0.5870       NA   0.9368
```

```r
OOSE.acc.rpart <- sum(predrpart == crossval$classe)/length(predrpart)
OOSE.rpart <- round(((1-OOSE.acc.rpart)*100), digits=2)
```
The Out of Sample Error using the Classification tree method was estimated to be 52.42%.

4c. Random Forest

```r
library(randomForest)
model <- randomForest(classe ~ ., data = train)
predCrossVal <- predict(model, crossval)
confusionMatrix(crossval$classe, predCrossVal) ## 99.4% accuracy
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1668    5    0    0    1
##          B    3 1135    1    0    0
##          C    0   10 1015    1    0
##          D    0    0   12  951    1
##          E    0    0    3    3 1076
## 
## Overall Statistics
##                                         
##                Accuracy : 0.993         
##                  95% CI : (0.991, 0.995)
##     No Information Rate : 0.284         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.991         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.998    0.987    0.984    0.996    0.998
## Specificity             0.999    0.999    0.998    0.997    0.999
## Pos Pred Value          0.996    0.996    0.989    0.987    0.994
## Neg Pred Value          0.999    0.997    0.997    0.999    1.000
## Prevalence              0.284    0.195    0.175    0.162    0.183
## Detection Rate          0.283    0.193    0.172    0.162    0.183
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.998    0.993    0.991    0.997    0.998
```

```r
OOSE.acc.rf <- sum(predCrossVal == crossval$classe)/length(predCrossVal)
OOSE.rf <- round(((1-OOSE.acc.rf)*100), digits=2)
```
The Out of Sample Error using the Random Forest method was estimated to be 0.68%.

As this was by far the most accurate of the 3 methods tested, and below the desired level of 5 %, I decided to use this model for the test data.

###Step 5. Import test data and tidy in the same manner as the training data

```r
testingraw <- read.csv("C:/Users/mikoflan/Documents/PredMachLearning/pml-testing.csv", 
                       na.strings= c("NA",""," "))
testingrawNA <- apply(testingraw, 2, function(x) sum(is.na(x)))

sum(testingrawNA > 10)
```

```
## [1] 100
```

```r
sum(testingrawNA > 15000)
```

```
## [1] 0
```

```r
sum(testingrawNA > 0) ## therefore can remove all cols with any NA (if one NA, then > 15000 NAs)
```

```
## [1] 100
```

```r
testing <- testingraw[,which(testingrawNA < 1)]

ncoltesting <- ncol(testing)
testing <- testing[,8:ncoltesting] ##remove non-predictive information in 1st 8 columns
```
###Step 6. Apply predictive model which gave best results in step 4.

```r
predicttesting <- predict(model, testing)
```
