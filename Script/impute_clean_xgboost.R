library(caTools)
library(caret)
library(xgboost)


train = read.csv("../Data/train_new.csv", stringsAsFactors = T)
test = read.csv("../Data/test_new.csv", stringsAsFactors = T)
rownames(train) = train$id
rownames(test) = test$id
train = train[,-1]
test = test[,-1]

train_price = train[,ncol(train)]

macro = read.csv("../Data/macro_impute.csv")

total = rbind(train[,-ncol(train)], test)
total = merge(total, macro, by = "timestamp")
total = total[,-1]
rownames(total) = c(rownames(train), rownames(test))

trainId = rownames(train)
testId = rownames(test)

colTypes = sapply(1:ncol(total), function(i) class(total[,i]))

total_factors = total[, colTypes == "factor"]

for(i in 1:ncol(total_factors)) {
  t = as.character(total_factors[,i], levels = unique(total_factors[,i]))
  t[is.na(t)] = "NA"
  total_factors[,i] = factor(t, levels = unique(t))
}

len_factor = apply(total_factors, 2, function(x) length(unique(x)))

total_factors$name =NULL

total_dummy = model.matrix(~0+., total_factors)

train = total[trainId, colTypes != "factor"]
train = cbind(train, total_dummy[trainId,])
train = cbind(train, price_doc = train_price)

test = total[testId, colTypes != "factor"]
test = cbind(test, total_dummy[testId,])

rm(total)
rm(total_dummy)
rm(total_factors)
rm(macro)

set.seed(88)
split = sample.split(train$price_doc, SplitRatio = 0.7)

datTest = train[!split,]
datTrain = train[split,]

ef = function(pred, real) {
  pred[is.na(pred)] = mean(real)
  pred[pred < 0] = min(real)
  return(sqrt(sum((log(pred+1) - log(real + 1))^2)/length(pred)))
}

model1 = xgboost(data = as.matrix(datTrain[,-ncol(datTrain)]),
                 label = datTrain[,ncol(datTrain)],
                 objective = "reg:linear",
                 eval_metric = "rmse", max_depth = 6, eta = 0.3, 
                 nthread = 8, nrounds = 1000, subsample = 0.7, colsample_bytree = 0.7)

# pred <- predict(model, dtest)

ef(predict(model1,  as.matrix(datTest)),  datTest$price_doc)

pred = predict(model1, as.matrix(test))

pred[pred < min(train$price_doc)] = min(train$price_doc)

write.csv(cbind(id = rownames(test),
                price_doc = pred),
          "impute_clean_xgboost_eval.csv", row.names = F)




lc = as.data.frame(matrix(0, nrow = 10, ncol = 3))
colnames(lc) = c("size", "TrainRMSE", "TestRMSE")

lc$size = ceiling(2.732^(1:10))


set.seed(5354)
rand_ind = sample(1:nrow(datTrain), nrow(datTrain))

for(i in 1:nrow(lc)) {
  temptrain = datTrain[rand_ind[1:lc[i,1]],]
  lcm = xgboost(data = as.matrix(temptrain[,-ncol(temptrain)]),
                label = temptrain[,ncol(temptrain)],
                objective = "reg:linear", silent = 1, nthread = 8,
                eval_metric = "rmse", nrounds = 500, max_depth = 5, 
                eta = 0.02, gamma = 0.1, colsample_bytree = 1, 
                min_child_weight = 1, subsample = 0.7)
  lc[i,2] = ef(predict(lcm, as.matrix(temptrain)),  temptrain$price_doc)
  lc[i,3] = ef(predict(lcm, as.matrix(datTest)),  datTest$price_doc)
}


ggplot(lc) + 
  geom_line(aes(x = size, y = TrainRMSE), color = "red") +
  geom_line(aes(x = size, y = TestRMSE), color = "blue") +
  geom_hline(yintercept = 0.32, color = "black")


# Tune

xgb_grid_1 = expand.grid(
  nrounds = 1000,
  eta = c(0.015, 0.02, 0.025),
  max_depth = c(4, 5, 6),
  gamma = c(3, 1, 0.3),
  colsample_bytree = c(0.75,0.85, 1), 
  min_child_weight = c(1, 2, 5),
  subsample = c(0.6, 0.7, 0.8)
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

save.image("./impute_clean_xgboost.RData")

head(xgb_train_1$results[order(xgb_train_1$results$RMSE),])

# eta max_depth gamma colsample_bytree min_child_weight subsample nrounds    RMSE  Rsquared   RMSESD
# 0.025         5   0.3             0.85                1       0.8    1000 2678919 0.7064678 70222.77
# 0.025         5   3.0             0.75                2       0.7    1000 2679136 0.7066265 60992.69
# 0.025         5   1.0             0.85                5       0.7    1000 2680768 0.7061018 58836.44
# 0.020         5   1.0             0.75                1       0.7    1000 2681452 0.7059229 65329.54
# 0.020         6   3.0             0.75                5       0.7    1000 2682107 0.7057659 50329.35
# 0.025         5   3.0             0.75                1       0.7    1000 2682915 0.7055868 62330.88
# RsquaredSD
# 0.01561210
# 0.01537002
# 0.01616134
# 0.01630266
# 0.01838025
# 0.01986486

## Best parameter
list(nrounds = 1000, max_depth = 5, eta = 0.025, gamma = 0.3, 
     colsample_bytree = 0.85, min_child_weight = 1, subsample = 0.88)



model2 = xgboost(data = as.matrix(datTrain[,-ncol(datTrain)]),
                 label = datTrain[,ncol(datTrain)],
                 objective = "reg:linear", nthread = 8,
                 eval_metric = "rmse", nrounds = 1000, max_depth = 5, 
                 eta = 0.025, gamma = 0.3, colsample_bytree = 0.85, 
                 min_child_weight = 1, subsample = 0.88)

# pred <- predict(model, dtest)

ef(predict(model2,  as.matrix(datTest)),  datTest$price_doc)

pred2 = predict(model2, as.matrix(test))

summary(pred2)

# pred2[pred2 < min(train$price_doc)] = min(train$price_doc)

write.csv(cbind(id = rownames(test),
                price_doc = pred2),
          "impute_clean_tune_xgboost_eval.csv", row.names = F)












