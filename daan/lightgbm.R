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
library(lightgbm)
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

# train_complete = train[complete.cases(train), ]
x_train = train %>% select(-isFraud)
y_train = train %>% select(isFraud) %>% unlist %>% as.factor() 

x_test = test %>% select(-isFraud)
y_test = test %>% select(isFraud) %>% unlist %>% as.factor()

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

y_train = train %>% select(isFraud) %>% unlist %>% as.numeric()-1
y_test = test %>% select(isFraud) %>% unlist %>% as.factor()

train_sparse = Matrix(as.matrix(train), sparse=TRUE)
test_sparse  = Matrix(as.matrix(test), sparse=TRUE)

lgb.train = lgb.Dataset(data=train_dummy, label=y_train)

# categoricals.vec = colnames(train)
lgb.grid = list(objective = "binary",
                metric = "auc",
                min_sum_hessian_in_leaf = 1,
                feature_fraction = 0.9,
                bagging_fraction = 1,
                bagging_freq = 1,
                max_bin = 255,
                lambda_l1 = 8,
                lambda_l2 = 1.3,
                min_data_in_bin=10,
                min_gain_to_split = 0,
                min_data_in_leaf = 10,
                is_unbalance = TRUE)

# Gini for Lgb
lgb.normalizedgini = function(preds, dtrain){
  actual = getinfo(dtrain, "label")
  score  = NormalizedGini(preds,actual)
  return(list(name = "gini", value = score, higher_better = TRUE))
}

# Train final model
lgb.model = lgb.train(params = lgb.grid, data = lgb.train, learning_rate = 0.01,
                      num_leaves = 25, num_threads = 2 , nrounds = 1000,
                      eval_freq = 20, eval = lgb.normalizedgini
                      #,categorical_feature = categoricals.vec
                      )
pred_out <- predict(lgb.model, test_dummy)
pred = ifelse(pred_out > 0.5, 1, 0)
pred = as.factor(pred)
confusionMatrix(pred, y_test)

pred = prediction(as.numeric(pred), as.numeric(y_test))
auc = performance(pred, measure = "auc")

perf = performance(pred,"tpr","fpr")

plot(perf, main = auc@y.values[[1]], colorize = TRUE)
