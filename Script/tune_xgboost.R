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


### 
# Learning Curve

efrmse = function(pred, real) {
  pred[is.na(pred)] = mean(real)
  pred[pred < 0] = min(real)
  return(sqrt(sum((pred - real)^2)/length(pred)))
}


lc = as.data.frame(matrix(0, nrow = 10, ncol = 3))
colnames(lc) = c("size", "TrainRMSE", "TestRMSE")

lc$size = ceiling(2.732^(1:10))


set.seed(934984)
rand_ind = sample(1:nrow(datTrain), nrow(datTrain))

for(i in 1:nrow(lc)) {
  temptrain = datTrain[rand_ind[1:lc[i,1]],]
  lcm = xgboost(data = as.matrix(temptrain[,-ncol(temptrain)]),
                 label = temptrain[,ncol(temptrain)],
                 objective = "reg:linear", silent = 1, nthread = 8,
                 eval_metric = "rmse", nrounds = 2000, max_depth = 5, 
                 eta = 0.02, gamma = 0.1, colsample_bytree = 1, 
                 min_child_weight = 1, subsample = 0.7)
  lc[i,2] = ef(predict(lcm, as.matrix(temptrain)),  temptrain$price_doc)
  lc[i,3] = ef(predict(lcm, as.matrix(datTest)),  datTest$price_doc)
}


ggplot(lc) + 
  geom_line(aes(x = size, y = TrainRMSE), color = "red") +
  geom_line(aes(x = size, y = TestRMSE), color = "blue") +
  geom_hline(yintercept = 0.32, color = "black")

### Reduced Feature Space ###

importance <- xgb.importance(feature_names = colnames(datTrain), model = lcm)

feat50 = head(importance$Feature, 50)

feat50_train = datTrain[,c(feat50, "price_doc")]
feat50_test = datTest[,c(feat50, "price_doc")]



lc50 = as.data.frame(matrix(0, nrow = 10, ncol = 3))
colnames(lc50) = c("size", "TrainRMSE", "TestRMSE")

lc50$size = ceiling(2.732^(1:10))


set.seed(934984)
rand_ind = sample(1:nrow(datTrain), nrow(datTrain))

for(i in 1:nrow(lc50)) {
  temptrain = feat50_train[rand_ind[1:lc50[i,1]],]
  lcm = xgboost(data = as.matrix(temptrain[,-ncol(temptrain)]),
                label = temptrain[,ncol(temptrain)],
                objective = "reg:linear", silent = 1, nthread = 8,
                eval_metric = "rmse", nrounds = 2000, max_depth = 5, 
                eta = 0.02, gamma = 0.1, colsample_bytree = 1, 
                min_child_weight = 1, subsample = 0.7)
  lc50[i,2] = ef(predict(lcm, as.matrix(temptrain)),  temptrain$price_doc)
  lc50[i,3] = ef(predict(lcm, as.matrix(feat50_test)),  feat50_test$price_doc)
}


ggplot(lc50) + 
  geom_line(aes(x = size, y = TrainRMSE), color = "red") +
  geom_line(aes(x = size, y = TestRMSE), color = "blue") +
  geom_hline(yintercept = 0.32, color = "black")


## Conclusion

## Underfitting

## Need more features



