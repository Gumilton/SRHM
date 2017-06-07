library(mice)

train = read.csv("../Data/train.csv", stringsAsFactors = T)
test = read.csv("../Data/test.csv", stringsAsFactors = T)
rownames(train) = train$id
rownames(test) = test$id
train = train[,-1]
test = test[,-1]

macro = read.csv("../Data/macro.csv", stringsAsFactors = T)
rownames(macro) = macro$timestamp
macro = macro[,-1]

md.pattern(macro)
colTypes = sapply(1:ncol(macro), function(i) class(macro[,i]))
imputed_Data <- mice(macro, m=5, maxit = 10, seed = 500,
                     method = ifelse(colTypes == "factor", "rf", "cart"))

ind_factor = which(colTypes == "factor")


boxplot(log2(colMeans(complete(imputed_Data, 1)[,-ind_factor], na.rm = T) - colMeans(macro[,-ind_factor], na.rm = T) + 300),
        log2(colMeans(complete(imputed_Data, 5)[,-ind_factor], na.rm = T) - colMeans(macro[,-ind_factor], na.rm = T) + 300))

macro_impute = complete(imputed_Data, 1)
macro_impute = cbind(timestamp = rownames(macro_impute),
                     macro_impute)

write.csv(macro_impute, "../Data/macro_impute.csv", row.names = F)
