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

colTypes = sapply(1:ncol(total), function(i) class(total[,i]))

total_factors = total[, colTypes == "factor"]

for(i in 1:ncol(total_factors)) {
  t = as.character(total_factors[,i], levels = unique(total_factors[,i]))
  t[is.na(t)] = "NA"
  total_factors[,i] = factor(t, levels = unique(t))
}

total_dummy = model.matrix(~0+., total_factors)

train = total[1:nrow(train), colTypes != "factor"]
train = cbind(train, total_dummy[1:nrow(train),])
train = cbind(train, price_doc = train_price)

test = total[-c(1:nrow(train)), colTypes != "factor"]
test = cbind(test, total_dummy[-c(1:nrow(train)),])

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
                 nthread = 2, nrounds = 1000, subsample = 0.7, colsample_bytree = 0.7)

# pred <- predict(model, dtest)

ef(predict(model1,  as.matrix(datTest)),  datTest$price_doc)

pred = predict(model1, as.matrix(test_noNA))
























