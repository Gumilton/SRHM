library(caTools)
library(caret)
library(xgboost)
library(mice)
library(VIM)


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

# sapply(ind_factor, function(x) table(total[,x]), simplify = T)

# Column 11 has too many levels, hard to converge for, remove from imputing.

# levely = 11

# aggr_plot <- aggr(total, col=c('navyblue','red'), numbers=TRUE, 
#                   sortVars=TRUE, labels=names(total), cex.axis=.7, 
#                   gap=3, ylab=c("Histogram of missing data","Pattern"))
# tempData <- mice(total[,-levely],m=5,maxit=50,meth='pmm',seed=500)
# summary(tempData)
# 
# completedData <- complete(tempData,1)



total_noNA = cbind(total_noNA[, -ind_factor],
              model.matrix(~0+., total_noNA[, ind_factor]))

# test$timestamp = factor(gsub("-([[:digit:]]{1,})$","",test$timestamp))

# train = train[,c(colnames(test), "price_doc")]

# ind_na = apply(train, 2, function(x) any(is.na(x)))

# train_noNA = train[,!ind_na]
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

dtrain <- xgb.DMatrix(data = as.matrix(datTrain[,-ncol(datTrain)]),
                      label = datTrain[,ncol(datTrain)])

dtest <- xgb.DMatrix(data = as.matrix(datTest[,-ncol(datTest)]),
                      label = datTest[,ncol(datTest)])

watchlist <- list(train=dtrain, test=dtest)
model1 = xgboost(data = as.matrix(datTrain[,-ncol(datTrain)]),
                label = datTrain[,ncol(datTrain)],
                objective = "reg:linear",
                eval_metric = "rmse", max_depth = 4, eta = 0.3, 
                nthread = 2, nrounds = 100)

model2 = xgboost(data = as.matrix(datTrain[,-ncol(datTrain)]),
                 label = datTrain[,ncol(datTrain)],
                 objective = "reg:linear",
                 eval_metric = "rmse", max_depth = c(10, 4), eta = 0.3, 
                 nthread = 2, nrounds = 10)

# pred <- predict(model, dtest)

ef(predict(model1, dtest),  datTest$price_doc)
ef(predict(model2, dtest),  datTest$price_doc)

pred = predict(model1, as.matrix(test_noNA))

write.csv(cbind(id = rownames(test_noNA),
                price_doc = pred),
          "noNA_xgboost_baseline_eval.csv", row.names = F)
