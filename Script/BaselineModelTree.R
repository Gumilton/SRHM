library(caTools)

library(rpart)



train = read.csv("../Data/train.csv")
rownames(train) = train$id
train = train[,-1]
# train$timestamp = factor(gsub("-([[:digit:]]{1,})$","",train$timestamp))


test = read.csv("../Data/test.csv")
# test$timestamp = factor(gsub("-([[:digit:]]{1,})$","",test$timestamp))

# train = train[,c(colnames(test), "price_doc")]

# ind_na = apply(train, 2, function(x) any(is.na(x)))

# train_noNA = train[,!ind_na]
train_noNA = train[,-1]
# test = test[,ind_na]
set.seed(88)
split = sample.split(train_noNA$price_doc, SplitRatio = 0.7)

datTest = train_noNA[!split,]
# datTest$timestamp = factor(datTest$timestamp, levels = levels(train$timestamp))
datTrain = train_noNA[split,]
# datTrain$timestamp = factor(datTrain$timestamp, levels = levels(train$timestamp))


tree = rpart(formula = price_doc ~ ., data=datTrain, 
             parms = list(split  = "information"), method="anova", 
             control = rpart.control(minsplit = 50))

treePrune<- prune(tree, cp= tree$cptable[which.min(tree$cptable[,"xerror"]),"CP"])

pred1 = predict(fit, datTest)
pred2 = predict(fit, datTest)

ef = function(pred, real) {
  pred[is.na(pred)] = mean(real)
  pred[pred < 0] = min(real)
  return(sqrt(sum((log(pred+1) - log(real + 1))^2)/length(pred)))
}

ef(predict(fit, datTest),  datTest$price_doc)

ef(predict(treePrune, datTest),  datTest$price_doc)

pred = predict(fit, test)
pred[is.na(pred)] = mean(train$price_doc)
pred[pred < 0] = min(train$price_doc)

write.csv(cbind(id = test$id,
                price_doc = pred),
          "baseline_tree_eval.csv", row.names = F)


pred = predict(treePrune, test)
pred[is.na(pred)] = mean(train$price_doc)
pred[pred < 0] = min(train$price_doc)

write.csv(cbind(id = test$id,
                price_doc = pred),
          "baseline_treePrune_eval.csv", row.names = F)

