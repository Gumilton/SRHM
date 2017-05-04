train = read.csv("../Data/train.csv")
rownames(train) = train$id
train = train[,-1]
test = read.csv("../Data/test.csv")
rownames(test) = test$id
test = test[,-1]

sm_train = summary(train)

