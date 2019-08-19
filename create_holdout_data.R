# clean environment
rm(list=ls())
gc(reset=TRUE)
cat('\014')

library(data.table)
library(dplyr)

train_transaction = fread("data/train_transaction.csv")
train_identity = fread("data/train_identity.csv")
combined = left_join(train_transaction, train_identity)
set.seed(1234)
train <- sample_n(combined, round(nrow(combined)/2), replace=FALSE)
holdout <- anti_join(combined, train)

fwrite(train, "data/train_dataset.csv")
fwrite(holdout, "data/holdout_dataset.csv")
