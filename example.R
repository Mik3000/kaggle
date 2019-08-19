# clean environment
rm(list=ls())
gc(reset=TRUE)
cat('\014')
setwd("~/Documents/kaggle")

library(data.table)
library(dplyr)
library(caret)
library(e1071)
library(randomForest)
library(ROCR)

train_transaction = fread("data/train_transaction.csv")
train_identity = fread("data/train_identity.csv")
train = left_join(train_transaction, train_identity)
rm(train_transaction, train_identity)

# drop columns with empty cells ("")
empty_cells = ifelse(train == '', 1, 0) %>% colSums()
train = train[, which(empty_cells == 0)]   

# drop columns with NA cells
missing_obs = apply(train, 2, function(col)sum(is.na(col))/length(col))
train = train[, missing_obs < 0.01]

#train = sample_n(train, 10000)
train$isFraud = train$isFraud %>% as.factor()

train %>% group_by(isFraud) %>% summarise(n = n()) %>% mutate(freq = n / sum(n))

x = train %>% select(-isFraud)
y = train %>% select(isFraud) %>% unlist()

# Naive bayes baseline
set.seed(1)
fit_nb = naiveBayes(x, y)
pred_nb = predict(fit_nb, newdata = x, type = "class")
confusionMatrix(pred_nb, y)
pred = prediction(as.numeric(pred_nb), as.numeric(y))
auc = performance(pred, measure = "auc")

plot(perf, main = auc@y.values[[1]], colorize = TRUE)
# RF 
train = train[complete.cases(train),]
colSums(is.na(train))

x = train %>% select(-isFraud)
y = train %>% select(isFraud) %>% unlist()

control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
seed <- 7
metric <- "Accuracy"
set.seed(seed)
mtry <- sqrt(ncol(x)) %>% round()
tunegrid <- expand.grid(.mtry = mtry)
rf_default <- train(x, y, method = "rf", metric = metric, tuneGrid = tunegrid, trControl = control)

function(x) {ifelse(x == '', 1, 0)}

apply(x, 2, function(x)  )
   
