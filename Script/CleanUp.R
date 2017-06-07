library(lubridate)


train = read.csv("../Data/train.csv")
test = read.csv("../Data/test.csv")
hist(log2(train$price_doc))
sm_train = summary(train)
price_log2 = log2(train$price_doc)

q1 = quantile(price_log2, .25)
q3 = quantile(price_log2, .75)

high = q3 + (q3-q1)*3
low = q1 - (q3-q1)*3

price_log2[price_log2 > high] = high
price_log2[price_log2 < low] = low

hist(log2(train$price_doc))
hist(price_log2)

total = rbind(train[,-ncol(train)], test)

total[which(total$life_sq > total$full_sq),"life_sq"] = NA

total[which(total$life_sq < 5),"life_sq"] = NA

total[which(total$full_sq < 5),"full_sq"] = NA

total[which(total$kitch_sq < 3),"kitch_sq"] = NA

total[which(total$full_sq > 1000), "full_sq"] = 53.26

total[which(total$full_sq > 150 & total$life_sq /total$full_sq < 0.3), "full_sq"] = NA

total[which(total$build_year > 2020), "build_year"] = NA

total[which(total$build_year < 1600), "build_year"] = NA

total[which(total$num_room == 0), "num_room"] = NA

total[which(total$floor == 0), "floor"] = NA

total[which(total$floor> total$max_floor), "max_floor"] = NA

total[which(total$state == 33), "state"] = NA

total$timestamp = as.Date(as.character(total$timestamp))

total$dayofyear =yday(total$timestamp)
total$dayofmonth =day(total$timestamp)
total$dayofweek = weekdays(total$timestamp)

total$weekofyear = week(total$timestamp)

total$month = months(total$timestamp)

total$rel_floor = total$floor / total$max_floor

total$rel_kitch_sq = total$kitch_sq / total$full_sq

total$rel_life_sq = total$life_sq / total$full_sq

total$name = paste(total$sub_area, total$metro_km_avto)

total$avg_room_sq = total$full_sq / total$num_room

train_new = total[1:nrow(train),]
train_new = cbind(train_new, 2^price_log2)
test_new = total[(nrow(train)+1):nrow(total),]


write.csv(train_new, "../Data/train_new.csv", row.names = F, quote = F)
write.csv(test_new, "../Data/test_new.csv", row.names = F, quote = F)
