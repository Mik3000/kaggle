xgbGrid <- expand.grid(nrounds = c(100,200),  # this is n_estimators in the python code above
                       max_depth = c(10, 15, 20, 25),
                       colsample_bytree = seq(0.5, 0.9, length.out = 5),
                       ## The values below are default values in the sklearn-api. 
                       eta = 0.1,
                       gamma=0,
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

xgb_base <- caret::train(
  x = x,
  y = y,
  trControl = xgb_trcontrol,
  tuneGrid = xgbGrid,
  method = "xgbTree",
  verbose = TRUE
)

xgb_base$bestTune

bstSparse <- xgboost(data = train_dummy, label = y, max_depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")
 
pred <- predict(bstSparse, train_dummy)
pred = ifelse(pred>0.5,1,0)
pred = as.factor(pred)
y = as.factor(y)
confusionMatrix(pred, y)
 
 