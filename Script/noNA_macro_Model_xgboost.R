library(caTools)
library(caret)
library(xgboost)


train = read.csv("../Data/train.csv", stringsAsFactors = T)
test = read.csv("../Data/test.csv", stringsAsFactors = T)
rownames(train) = train$id
rownames(test) = test$id
train = train[,-1]
test = test[,-1]

macro = read.csv("../Data/macro.csv", stringsAsFactors = T)

train = merge(train, macro, by = "timestamp")
train = train[,c(1:290, 292:ncol(train), 291)]
test = merge(test, macro, by = "timestamp")

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

model1 = xgboost(data = as.matrix(datTrain[,-ncol(datTrain)]),
                 label = datTrain[,ncol(datTrain)],
                 objective = "reg:linear",
                 eval_metric = "rmse", max_depth = 6, eta = 0.3, 
                 nthread = 8, nrounds = 1000)

# pred <- predict(model, dtest)

ef(predict(model1,  as.matrix(datTest)),  datTest$price_doc)

pred = predict(model1, as.matrix(test_noNA))

write.csv(cbind(id = rownames(test_noNA),
                price_doc = pred),
          "noNA_xgboost_macro_beforeTune_eval.csv", row.names = F)


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

### Tune