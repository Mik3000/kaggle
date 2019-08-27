# clean environment
rm(list=ls())
gc(reset=TRUE)
cat('\014')
setwd("~/kaggle")

library(data.table)
library(dplyr)
library(caret)
library(ROCR)
library(purrr)
library(xgboost)
library(Matrix)

train_transaction = fread("data/train_transaction.csv", na.strings=c("","NA"), stringsAsFactors = TRUE)
train_identity = fread("data/train_identity.csv", na.strings=c("","NA"), stringsAsFactors = TRUE)
dataset = left_join(train_transaction, train_identity)
rm(train_transaction, train_identity)

# train on subset
set.seed(1234)
dataset = sample_n(dataset, 10000)

dataset = map_if(dataset, is.character, as.factor) %>% as.data.frame()

ind <- createDataPartition(dataset$isFraud, p = .7, list = FALSE)
train <- dataset[ind, ]
test  <- dataset[-ind, ]

#train_complete = train[complete.cases(train), ]
x_train = train %>% select(-isFraud)
y_train = train %>% select(isFraud) %>% unlist %>% as.factor()

# upsample
train = upSample(x = x_train, y = y_train, yname = "isFraud")

options(na.action='na.pass')
train_dummy = sparse.model.matrix(isFraud ~ . -1, train)
#saveRDS(train_dummy, "data/train_dummy.RDS")
test_dummy = sparse.model.matrix(isFraud ~ . -1, test)
#saveRDS(test_dummy, "data/test_dummy.RDS")

# drop columns with NA cells
#missing_obs = apply(train, 2, function(col) sum(is.na(col)) / length(col))
#train = train[, missing_obs < 0.01]

test %>% group_by(isFraud) %>% summarise(n = n()) %>% mutate(freq = n / sum(n))

y_train = train %>% select(isFraud) %>% unlist %>% as.numeric()-1
y_test = test %>% select(isFraud) %>% unlist %>% as.factor()

eta = c(0.05, 0.1, 0.2, 0.4, 0.5, 0.7, 1)
n_settings = length(eta)
recall_out = rep(NA, n_settings)
for (i in 1:n_settings){
  xgb_fit = xgboost(data = train_dummy, label = y_train, max_depth = 2, 
                    eta = eta[i], nthread = 2, nrounds = 10, objective = "binary:logistic")
  pred_out <- predict(xgb_fit, test_dummy)
  pred = ifelse(pred_out > 0.5, 1, 0)
  pred = as.factor(pred)
  conf = confusionMatrix(pred, y_test)
  recall = conf$byClass['Specificity'] %>% unlist(use.names = FALSE)
  recall_out[i] = recall
}
max_eta = eta[which.max(recall_out)]

xgb_fit = xgboost(data = train_dummy, label = y_train, max_depth = 2, 
                  eta = eta, nthread = 2, nrounds = 10, objective = "binary:logistic")

pred_out <- predict(xgb_fit, train_dummy)
pred = ifelse(pred_out > 0.5, 1, 0)
pred = as.factor(pred)
y_train_factor = as.factor(y_train)
confusionMatrix(pred, y_train_factor)

pred_out <- predict(xgb_fit, test_dummy)
pred = ifelse(pred_out > 0.5, 1, 0)
pred = as.factor(pred)
confusionMatrix(pred, y_test)

pred = prediction(as.numeric(pred), as.numeric(y_test))
auc = performance(pred, measure = "auc")

perf = performance(pred,"tpr","fpr")

plot(perf, main = auc@y.values[[1]], colorize = TRUE)
