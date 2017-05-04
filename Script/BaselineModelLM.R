library(caTools)

library(rpart)



train = read.csv("../Data/train.csv")
rownames(train) = train$id
train = train[,-1]
train$timestamp = factor(gsub("-([[:digit:]]{1,})$","",train$timestamp))


test = read.csv("../Data/test.csv")
test$timestamp = factor(gsub("-([[:digit:]]{1,})$","",test$timestamp))

# train = train[,c(colnames(test), "price_doc")]

ind_na = apply(train, 2, function(x) any(is.na(x)))

train_noNA = train[,!ind_na]
train_noNA = train_noNA[,-1]
# test = test[,ind_na]
set.seed(88)
split = sample.split(train_noNA$price_doc, SplitRatio = 0.7)

datTest = train_noNA[!split,]
# datTest$timestamp = factor(datTest$timestamp, levels = levels(train$timestamp))
datTrain = train_noNA[split,]
# datTrain$timestamp = factor(datTrain$timestamp, levels = levels(train$timestamp))


fit <- glm(price_doc ~ .,data=datTrain)
sf = summary(fit)

# feat = c("full_sq", "product_type")

pred1 = predict(fit, datTest)
pred2 = predict(fit, datTest)



pred = predict(fit, test)
pred[is.na(pred)] = 0
pred[pred < 0] = 0

write.csv(cbind(id = test$id,
                price_doc = pred),
          "baseline_eval.csv", row.names = F)
