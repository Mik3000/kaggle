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

train_transaction = fread("data/train_transaction.csv")
train_identity = fread("data/train_identity.csv")
train = left_join(train_transaction, train_identity)
rm(train_transaction, train_identity)

# train on subset
set.seed(1234)
train = sample_n(train, 50000)

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

param <-  data.frame(nrounds=c(100), max_depth = c(2),eta =c(0.3),gamma=c(0),
                     colsample_bytree=c(0.8),min_child_weight=c(1),subsample=c(1)) 
xgb_fit <- train(isFraud ~ ., train, method = "xgbTree", metric = "Accuracy", trControl = trainControl(method="none"), tuneGrid = param)
pred_xgb <- predict(xgb_fit, newdata = x)

confusionMatrix(pred_xgb, y)
pred = prediction(as.numeric(pred_xgb), as.numeric(y))
auc = performance(pred, measure = "auc")

perf = performance(pred,"tpr","fpr")

plot(perf, main = auc@y.values[[1]], colorize = TRUE)

xgb_imp = varImp(xgb_fit)
plot(xgb_imp)
