# clean environment
rm(list=ls())
gc(reset=TRUE)
cat('\014')
setwd("~/kaggle")

library(data.table)
library(dplyr)
library(caret)
library(e1071)
library(randomForest)
library(ROCR)
library(purrr)
library(xgboost)
library(Matrix)

train_transaction = fread("data/train_transaction.csv", na.strings=c("","NA"), stringsAsFactors = TRUE)
train_identity = fread("data/train_identity.csv", na.strings=c("","NA"), stringsAsFactors = TRUE)
train = left_join(train_transaction, train_identity)
rm(train_transaction, train_identity)

# train on subset
set.seed(1234)
train = sample_n(train, 10000)

train = map_if(train, is.character, as.factor) %>% as.data.frame()

#train_complete = train[complete.cases(train), ]

options(na.action='na.pass')
train_dummy = sparse.model.matrix(isFraud ~ . -1, train)

# drop columns with NA cells
#missing_obs = apply(train, 2, function(col) sum(is.na(col)) / length(col))
#train = train[, missing_obs < 0.01]

train %>% group_by(isFraud) %>% summarise(n = n()) %>% mutate(freq = n / sum(n))

x = train_dummy[ ,1:100]
y = train %>% select(isFraud) %>% unlist %>% as.integer()-1


xgbGrid <- expand.grid(nrounds = c(100, 200),
                       max_depth = c(10, 15, 20, 25),
                       colsample_bytree = seq(0.5, 0.9, length.out = 5),
                       eta = 0.1,
                       gamma = 0,
                       min_child_weight = 1,
                       subsample = 1
)

xgb_trcontrol = trainControl(
  method = "cv",
  number = 5,  
  allowParallel = FALSE,
  verboseIter = FALSE,
  returnData = FALSE
)

xgb_fit = xgboost(data = train_dummy, label = y, max_depth = 2, 
                  eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")

pred <- predict(xgb_fit, train_dummy)
pred = ifelse(pred > 0.5, 1, 0)
pred = as.factor(pred)
act = as.factor(y)
confusionMatrix(pred, act)

pred = prediction(as.numeric(pred), as.numeric(y))
auc = performance(pred, measure = "auc")

perf = performance(pred,"tpr","fpr")

plot(perf, main = auc@y.values[[1]], colorize = TRUE)

xgb_imp = varImp(xgb_fit)
plot(xgb_imp)
xgb.plot.importance(xgb_fit)
