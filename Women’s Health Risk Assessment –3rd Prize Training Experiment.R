# Import library xgboost.
# Please keep in mind, when you implement your model in Azure ML, you need to also import this 
# library in your Execute R Script module.
# Otherwise, you will encounter error when you use the model to predict

library(xgboost)
library(dplyr)

# Specify the URL of data. 
# Also specify the rda file that you want to use to save the model
dataURL <- 'http://az754797.vo.msecnd.net/competition/whra/data/WomenHealth_Training.csv'

religions <- c("0", "Buddhist", "Evangelical/Bo", "Hindu", "Jewish",
  "Muslim", "Other", "Other Christia", "Roman Catholic", "Russian/Easter",
  "Traditional/An")
religions.code <- c(0, 1, 2, 3, 4, 5, 0, 6, 7, 8, 9)

df.religion <- data.frame(religion = religions, Intreligion = religions.code)

# Read data to R workspace. The string field religion is read as factors
colclasses <- rep("integer", 50)
colclasses[36] <- "character"
dataset1 <- read.table(dataURL, header = TRUE, sep = ",", strip.white = TRUE,
  stringsAsFactors = F, colClasses = colclasses)
summary(dataset1)

# Combine columns geo, segment, and subgroup into a single column. 
# This will be the label column to be predicted in the multiclass classification task
combined_label <- 100 * dataset1$geo + 10 * dataset1$segment + dataset1$subgroup
data.set <- cbind(dataset1, combined_label)
data.set$combined_label <- as.factor(data.set$combined_label)

# Clean missing data by replacing missing values with -1 (with "0" for string variable religion)
data.set[is.na(data.set)] <- -1
data.set[data.set$religion == "", "religion"] <- "0"
data.set$religion <- factor(data.set$religion)
data.set$combined_label <- relevel(data.set$combined_label, ref = '111')
data.set$combined_label <- as.numeric(as.factor(data.set$combined_label)) - 1

data.set <- left_join(data.set, df.religion, by = c("religion"))

summary(data.set)
names(data.set)

# Skip the columns patientID, segment, subgroup, and INTNR from the feature set
ncols <- ncol(data.set)
feature_index <- c(2:18, 20:35, 37:48, 52)

# Split the data into train and validation data
nrows <- nrow(data.set)
sample_size <- floor(0.75 * nrows)
set.seed(1234) # set the seed to make your partition reproductible
train_ind <- sample(seq_len(nrows), size = sample_size)

train <- data.set[train_ind,]
validation <- data.set[ - train_ind,]

x <- data.matrix(train[, feature_index])
y <- train$combined_label

xx <- xgb.DMatrix(x, label = train$combined_label)

newx <- data.matrix(validation[, feature_index])
newxx <- xgb.DMatrix(newx, label = validation$combined_label)

modelGLM.xgb <- xgb.train(data = xx,
                  #label = y,
                  nrounds = 400,
                  watchlist = list(train = xx, validation = newxx),
                  eta = 0.01,
                  max_depth = 12,
                  subsample = 0.9,
                  colsample_bytree = 0.9,
                  objective = "multi:softprob",
                  num_class = 37,
                  eval_metric = "merror",
                  set.seed = 52,
                  nthread = 4)

# Predict the validation data
predicted <- predict(modelGLM.xgb, newxx)

# reshape it to a num_class-columns matrix
pred <- matrix(predicted, ncol=37, byrow=TRUE)
# convert the probabilities to softmax labels
pred_labels <- max.col(pred) - 1

# predictedMatrix <- matrix(predicted, ncol = 37, byrow = T)

# df <- data.frame(real=validation$combined_label, predicho=pred_labels)
# confMXGB <- caret::confusionMatrix(predicted, validation$combined_label)
# confMXGB$table

accuracy <- round(sum(pred_labels == validation$combined_label) / nrow(validation) * 100, 6)
print(paste("The accuracy on validation data is ", accuracy, "%", sep = ""))

# Save the model in rda file
model_rda_file <- "C:/Users/david/Documents/Training/Cortana Competitions/WomenRisk/xgbmodel_DATE_VERSION.rda"
save(modelGLM.xgb, file = model_rda_file)