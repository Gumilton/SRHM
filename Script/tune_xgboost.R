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
  nrounds = c(100, 1000),
  eta = c(0.5, 0.1, 0.01, 0.001),
  max_depth = c(2, 5, 7, 10),
  gamma = c(10, 3, 1, 0.3, 0.1, 0.01),
  colsample_bytree = c(0.5, 0.7, 0.9, 1), 
  min_child_weight = seq(1, 11, 4),
  subsample = seq(0.4, 1, 0.3)
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
