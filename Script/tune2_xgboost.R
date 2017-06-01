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
  eta = c(0.01, 0.02, 0.05),
  max_depth = c(4, 5, 6),
  gamma = c(3, 1, 0.3, 0.1),
  colsample_bytree = c(0.75, 1), 
  min_child_weight = c(1,5),
  subsample = c(0.7, 0.8)
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

save.image("./tune2_xgboost.RData")

## Best parameter
list(nrounds = 1000, max_depth = 6, 
     eta = 0.01, gamma = 1, colsample_bytree = 0.75,
     min_child_weight = 1, subsample = 0.7)

head(xgb_train_1$results[order(xgb_train_1$results$RMSE),])

# eta max_depth gamma colsample_bytree min_child_weight subsample nrounds    RMSE  Rsquared   RMSESD RsquaredSD
# 0.01         6   1.0             0.75                1       0.7    1000 2794886 0.6803368 278601.1 0.03899312
# 0.01         6   0.1             0.75                5       0.7    1000 2794979 0.6800665 294633.3 0.04346798
# 0.01         6   3.0             0.75                5       0.7    1000 2795363 0.6801428 291938.0 0.04308585
# 0.01         5   0.1             1.00                1       0.7    1000 2795471 0.6801262 284079.4 0.04085406
# 0.01         5   1.0             0.75                5       0.8    1000 2795594 0.6802320 290852.8 0.04164412
# 0.01         6   0.3             0.75                5       0.7    1000 2796325 0.6799031 284956.1 0.04027388


model2 = xgboost(data = as.matrix(datTrain[,-ncol(datTrain)]),
                 label = datTrain[,ncol(datTrain)],
                 objective = "reg:linear", silent = 2, nthread = 8,
                 eval_metric = "rmse", nrounds = 1000, max_depth = 6, 
                 eta = 0.01, gamma = 1, colsample_bytree = 0.75,
                 min_child_weight = 1, subsample = 0.7)
ef(predict(model2, as.matrix(datTest)),  datTest$price_doc)

pred = predict(model2, as.matrix(test_noNA))

write.csv(cbind(id = rownames(test_noNA),
                price_doc = pred),
          "noNA_xgboost_tune2_eval.csv", row.names = F)


