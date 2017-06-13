library(caTools)
library(caret)
# library(FNN)
# library(KernelKnn)
# library(e1071)
library(doSNOW)
library(foreach)
library(dplyr)
library(ggplot2)

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
# col_na = apply(train, 2, anyNA)
# na_mean = colMeans(train[,col_na], na.rm = T)

for(i in 1:ncol(train)) {
  na_row = is.na(train[,i])
  if(sum(na_row) > 0) {
    train[na_row, i] = mean(train[,i], na.rm = T)
  }
}

train = cbind(train, total_dummy[trainId,])
train = cbind(train, price_doc = train_price)

test = total[testId, colTypes != "factor"]

for(i in 1:ncol(test)) {
  na_row = is.na(test[,i])
  if(sum(na_row) > 0) {
    test[na_row, i] = mean(train[,i], na.rm = T)
  }
}

test = cbind(test, total_dummy[testId,])

stopifnot(!(anyNA(test) | anyNA(train)))

# rm(total)
# rm(total_dummy)
# rm(total_factors)
# rm(macro)

set.seed(88)
split = sample.split(train$price_doc, SplitRatio = 0.7)

datTest = train[!split,]
datTrain = train[split,]


cl = makeCluster(6)
registerDoSNOW(cl)


ef = function(pred, real) {
  pred[is.na(pred)] = mean(real)
  pred[pred < 0] = min(real)
  return(sqrt(sum((log(pred+1) - log(real + 1))^2)/length(pred)))
}


# fit_cv_pair1 = KernelKnnCV(data = datTrain[,-ncol(datTrain)],
#                            y = datTrain[,ncol(datTrain)], k = 10 ,
#                            folds = 3, regression = T, 
#                            threads = 1)

# cv_knn = tune(knnreg, train.x = datTrain[,-ncol(datTrain)],
#               train.y = datTrain[,ncol(datTrain)], k = seq(2, 20, 2),
#               tunecontrol=tune.control(sampling = "cross", cross=3))
# 
# fit_knn = knnreg(x = datTrain[,-ncol(datTrain)], y = datTrain[,ncol(datTrain)], k = 10)
# 
# predict(fit_knn, datTest)

set.seed(847)

split1 = sample.split(datTrain$price_doc, SplitRatio = 0.67)
split2 = sample.split(datTrain$price_doc[!split1], SplitRatio = 0.5)


folds= list()
folds[[1]] = datTrain[!split1,]
folds[[2]] = datTrain[split1,][split2,]
folds[[3]] = datTrain[split1,][!split2,]

cv_grid = expand.grid(fold = 1:3,
                      k = seq(4,30,2))

cv_res = foreach(i= 1:nrow(cv_grid), .packages=c('caret'),
                 .verbose = T, .combine = rbind) %dopar%  {
                   
                   f = cv_grid[i,1]
                   ck = cv_grid[i,2]
                   vd = folds[[f]]
                   td = rbind(folds[[ifelse(f+1 > 3, (f+1)%%3, f+1)]],
                              folds[[ifelse(f+2 > 3, (f+2)%%3, f+2)]])
                   
                   fit = knnreg(x = td[,-ncol(td)], y =  td[,ncol(td)], k = ck)
                   error = ef(predict(fit, vd[,-ncol(vd)]), vd[,ncol(vd)])
                   
                   return(c(f, ck, error))
}

colnames(cv_res) = c("Fold", "k", "error")

cv_summary = as.data.frame(cv_res) %>%
  group_by(k) %>%
  summarise(mean = mean(error), sd = sd(error))

ggplot(cv_summary, aes(x = k, y = mean)) + geom_line() + geom_point()

save.image("./impute_clean_knn.RData")

bestk = 16

fit = knnreg(x = datTrain[,-ncol(datTrain)], y =  datTrain[,ncol(datTrain)], k = bestk)
error = ef(predict(fit, datTest[,-ncol(datTest)]), datTest[,ncol(datTest)])

pred = predict(fit, test)

write.csv(cbind(id = rownames(test),
                price_doc = pred),
          "impute_clean_tune_knn_k16_eval.csv", row.names = F)


