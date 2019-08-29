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
library(tidyr)

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
p_upsample = 0.3
n_resample = ((length(y_train[y_train == 0])/ (1-p_upsample)) * p_upsample) %>% round()
resampled = sample_n(x_train[y_train  == 1,], n_resample, replace = TRUE)
resampled$isFraud = 1
train = data.frame(x_train, isFraud = y_train) %>% rbind(resampled)

options(na.action = 'na.pass')
train_dummy = sparse.model.matrix(isFraud ~ . -1, train)
#saveRDS(train_dummy, "data/train_dummy.RDS")
test_dummy = sparse.model.matrix(isFraud ~ . -1, test)
#saveRDS(test_dummy, "data/test_dummy.RDS")

# drop columns with NA cells
#missing_obs = apply(train, 2, function(col) sum(is.na(col)) / length(col))
#train = train[, missing_obs < 0.01]

train %>% group_by(isFraud) %>% summarise(n = n()) %>% mutate(freq = n / sum(n))

y_train = train %>% select(isFraud) %>% unlist %>% as.numeric()-1
y_test = test %>% select(isFraud) %>% unlist %>% as.factor()


### TUNE ETA ###
eta = c(0.05, 0.1, 0.2, 0.4, 0.5, 0.7, 1)
n_settings = length(eta)
n_rounds = 200
recall_out = rep(NA, n_settings)
conv_error = matrix(nrow = n_settings, ncol = n_rounds)
for (i in 1:n_settings){
  params = list(eta = eta[i], colsample_bylevel= 2/3,
              subsample = 0.5, max_depth = 2,
              min_child_weigth = 1)
  xgb_fit = xgboost(train_dummy, label = y_train, nrounds = n_rounds, params = params, 
                    objective = "binary:logistic", eval_metric = 'auc')
  conv_error[i,] = xgb_fit$evaluation_log$train_auc
  pred_out = predict(xgb_fit, test_dummy)
  pred = ifelse(pred_out > 0.5, 1, 0)
  pred = as.factor(pred)
  conf = confusionMatrix(pred, y_test)
  recall = conf$byClass['Specificity'] %>% unlist(use.names = FALSE)
  recall_out[i] = recall
}

max_eta = eta[which.max(recall_out)]

colnames(conv_error) = 1:n_rounds
conv_error = cbind(conv_error, eta) %>% as.data.frame()
conv_error = conv_error %>% gather(round, error, -eta)
conv_error$eta = as.character(conv_error$eta)
conv_error$round = as.numeric(conv_error$round)
ggplot(conv_error) + geom_line(aes(x = round, y = error, color = eta))

params = list(eta = max_eta, colsample_bylevel= 2/3,
              subsample = 0.5, max_depth = 2,
              min_child_weigth = 1)

xgb_fit = xgboost(train_dummy, label = y_train, nrounds = 200, params = params, 
                  objective = "binary:logistic", eval_metric = 'auc')

### colsample ###
colsample = c(1/3,2/3,1)
n_settings = length(colsample)
n_rounds = 200
recall_out = rep(NA, n_settings)
conv_error = matrix(nrow = n_settings, ncol = n_rounds)
for (i in 1:n_settings){
  params = list(eta = max_eta, colsample_bylevel= colsample[i],
                subsample = 0.5, max_depth = 2,
                min_child_weigth = 1)
  xgb_fit = xgboost(train_dummy, label = y_train, nrounds = n_rounds, params = params, 
                    objective = "binary:logistic", eval_metric = 'auc')
  conv_error[i,] = xgb_fit$evaluation_log$train_auc
  pred_out = predict(xgb_fit, test_dummy)
  pred = ifelse(pred_out > 0.5, 1, 0)
  pred = as.factor(pred)
  conf = confusionMatrix(pred, y_test)
  recall = conf$byClass['Specificity'] %>% unlist(use.names = FALSE)
  recall_out[i] = recall
}

max_colsample = colsample[which.max(recall_out)]

colnames(conv_error) = 1:n_rounds
conv_error = cbind(conv_error, colsample) %>% as.data.frame()
conv_error = conv_error %>% gather(round, error, -colsample)
conv_error$colsample = as.character(conv_error$colsample)
conv_error$round = as.numeric(conv_error$round)
ggplot(conv_error) + geom_line(aes(x = round, y = error, color = colsample))

params = list(eta = max_eta, colsample_bylevel = max_colsample,
              subsample = 0.5, max_depth = 2,
              min_child_weigth = 1)

xgb_fit = xgboost(train_dummy, label = y_train, nrounds = 200, params = params, 
                  objective = "binary:logistic", eval_metric = 'auc')

### Max depth
maxdepth = c(1, 2,4,6,10)
n_settings = length(maxdepth)
n_rounds = 200
recall_out = rep(NA, n_settings)
conv_error = matrix(nrow = n_settings, ncol = n_rounds)
for (i in 1:n_settings){
  params = list(eta = max_eta, colsample_bylevel = max_colsample,
                subsample = 0.5, max_depth = maxdepth[i],
                min_child_weigth = 1)
  xgb_fit = xgboost(train_dummy, label = y_train, nrounds = n_rounds, params = params, 
                    objective = "binary:logistic", eval_metric = 'auc')
  conv_error[i,] = xgb_fit$evaluation_log$train_auc
  pred_out = predict(xgb_fit, test_dummy)
  pred = ifelse(pred_out > 0.5, 1, 0)
  pred = as.factor(pred)
  conf = confusionMatrix(pred, y_test)
  recall = conf$byClass['Specificity'] %>% unlist(use.names = FALSE)
  recall_out[i] = recall
}

max_maxdepth = maxdepth[which.max(recall_out)]

colnames(conv_error) = 1:n_rounds
conv_error = cbind(conv_error, maxdepth) %>% as.data.frame()
conv_error = conv_error %>% gather(round, error, -maxdepth)
conv_error$maxdepth = as.character(conv_error$maxdepth)
conv_error$round = as.numeric(conv_error$round)
ggplot(conv_error) + geom_line(aes(x = round, y = error, color = maxdepth))

params = list(eta = max_eta, colsample_bylevel = max_colsample,
              subsample = 0.5, max_depth = max_maxdepth,
              min_child_weigth = 1)

xgb_fit = xgboost(train_dummy, label = y_train, nrounds = 200, params = params, 
                  objective = "binary:logistic", eval_metric = 'auc')

# tune subsample
subsample = c(0.25, 0.5, 0.75, 1)
n_settings = length(subsample)
n_rounds = 200
recall_out = rep(NA, n_settings)
conv_error = matrix(nrow = n_settings, ncol = n_rounds)
for (i in 1:n_settings){
  params = list(eta = max_eta, colsample_bylevel = max_colsample,
                subsample = subsample[1], max_depth = max_maxdepth,
                min_child_weigth = 1)
  xgb_fit = xgboost(train_dummy, label = y_train, nrounds = n_rounds, params = params, 
                    objective = "binary:logistic", eval_metric = 'auc')
  conv_error[i,] = xgb_fit$evaluation_log$train_auc
  pred_out = predict(xgb_fit, test_dummy)
  pred = ifelse(pred_out > 0.5, 1, 0)
  pred = as.factor(pred)
  conf = confusionMatrix(pred, y_test)
  recall = conf$byClass['Specificity'] %>% unlist(use.names = FALSE)
  recall_out[i] = recall
}

max_subsample = subsample[which.max(recall_out)]

colnames(conv_error) = 1:n_rounds
conv_error = cbind(conv_error, subsample) %>% as.data.frame()
conv_error = conv_error %>% gather(round, error, -subsample)
conv_error$subsample = as.character(conv_error$subsample)
conv_error$round = as.numeric(conv_error$round)
ggplot(conv_error) + geom_line(aes(x = round, y = error, color = subsample))

params = list(eta = max_eta, colsample_bylevel = max_colsample,
              subsample = max_subsample, max_depth = max_maxdepth,
              min_child_weigth = 1)

xgb_fit = xgboost(train_dummy, label = y_train, nrounds = 200, params = params, 
                  objective = "binary:logistic", eval_metric = 'auc')

### Validate performance
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
plot(perf, main = paste0("AUC: ", auc@y.values[[1]]), colorize = TRUE)


