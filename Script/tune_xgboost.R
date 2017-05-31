library(caTools)
library(caret)
library(xgboost)
# library(mice)
# library(VIM)


train = read.csv("../Data/train.csv", stringsAsFactors = T)
test = read.csv("../Data/test.csv", stringsAsFactors = T)
rownames(train) = train$id
rownames(test) = test$id
train = train[,-1]
test = test[,-1]

## no time stamp
train = train[,-1]
test = test[,-1]

total = rbind(train[,-ncol(train)], test)

colNA = apply(total,2, anyNA)

## NO NA Model

total_noNA = total[,!colNA]

colTypes = sapply(1:ncol(total_noNA), function(i) class(total_noNA[,i]))

ind_factor = which(colTypes == "factor")


total_noNA = cbind(total_noNA[, -ind_factor],
                   model.matrix(~0+., total_noNA[, ind_factor]))

train_noNA = total_noNA[rownames(train),]
train_noNA$price_doc = train$price_doc
test_noNA = total_noNA[rownames(test),]
# test = test[,ind_na]
set.seed(88)
split = sample.split(train_noNA$price_doc, SplitRatio = 0.7)

datTest = train_noNA[!split,]
# datTest$timestamp = factor(datTest$timestamp, levels = levels(train$timestamp))
datTrain = train_noNA[split,]
# datTrain$timestamp = factor(datTrain$timestamp, levels = levels(train$timestamp))

ef = function(pred, real) {
  pred[is.na(pred)] = mean(real)
  pred[pred < 0] = min(real)
  return(sqrt(sum((log(pred+1) - log(real + 1))^2)/length(pred)))
}


### Tune

xgb_grid_1 = expand.grid(
  nrounds = 1000,
  eta = c(0.5, 0.1, 0.02),
  max_depth = c(2, 5, 8),
  gamma = c(3, 1, 0.3, 0.1),
  colsample_bytree = c(0.5, 0.75, 1), 
  min_child_weight = seq(1, 11, 4),
  subsample = c(0.7, 1)
)

xgb_trcontrol_1 = trainControl(
  method = "cv",
  number = 3,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "all",                   # save losses across all models
  classProbs = TRUE,                      # set to TRUE for AUC to be computed
  allowParallel = TRUE
)

xgb_train_1 = train(
  x = datTrain[,-ncol(datTrain)],
  y = datTrain[,ncol(datTrain)],
  trControl = xgb_trcontrol_1,
  tuneGrid = xgb_grid_1,
  method = "xgbTree",
  eval_metric = "rmse",
  objective = "reg:linear",
  silent = 1,
  nthread = 8
)

save.image("./tune_xgboost.RData")

## Best parameter
list(nrounds = 1000, max_depth = 5, 
     eta = 0.02, gamma = 0.1, 
     colsample_bytree = 1, min_child_weight = 1, subsample = 0.7)

head(xgb_train_1$results[order(xgb_train_1$results$RMSE),])

# eta max_depth gamma colsample_bytree min_child_weight subsample nrounds    RMSE  Rsquared   RMSESD RsquaredSD
# 0.02         5   0.1             1.00                1       0.7    1000 2802205 0.6786370 333706.4 0.04941481
# 0.02         5   3.0             0.50                5       0.7    1000 2805555 0.6780519 326617.1 0.04690415
# 0.02         5   3.0             1.00                1       0.7    1000 2805825 0.6779681 323403.4 0.04596495
# 0.02         5   0.3             0.75                1       0.7    1000 2806397 0.6778889 315824.2 0.04359049
# 0.02         5   0.3             0.75                9       0.7    1000 2806753 0.6777403 323940.4 0.04569755
# 0.02         5   1.0             1.00                5       0.7    1000 2807083 0.6775499 336624.3 0.04941697


model1 = xgboost(data = as.matrix(datTrain[,-ncol(datTrain)]),
                 label = datTrain[,ncol(datTrain)],
                 objective = "reg:linear", silent = 2, nthread = 8,
                 eval_metric = "rmse", nrounds = 1000, max_depth = 5, 
                 eta = 0.02, gamma = 0.1, 
                 colsample_bytree = 1, min_child_weight = 1, subsample = 0.7)
ef(predict(model1, as.matrix(datTest)),  datTest$price_doc)

pred = predict(model1, as.matrix(test_noNA))

write.csv(cbind(id = rownames(test_noNA),
                price_doc = pred),
          "noNA_xgboost_tune1_eval.csv", row.names = F)




model2 = xgboost(data = as.matrix(datTrain[,-ncol(datTrain)]),
                 label = datTrain[,ncol(datTrain)],
                 objective = "reg:linear", silent = 2, nthread = 8,
                 eval_metric = "rmse", nrounds = 1000, max_depth = 5, 
                 eta = 0.02, gamma = 3, 
                 colsample_bytree = 0.5, min_child_weight = 5, subsample = 0.7)
ef(predict(model2, as.matrix(datTest)),  datTest$price_doc)

pred = predict(model2, as.matrix(test_noNA))

write.csv(cbind(id = rownames(test_noNA),
                price_doc = pred),
          "noNA_xgboost_tune2_eval.csv", row.names = F)

