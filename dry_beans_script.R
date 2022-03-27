
library(readxl)
library(ggplot2)
library(dplyr)

print("Reading in the Excel file...")
drybeans <- read_excel("Dry_Bean_Dataset.xlsx")

table(drybeans$Class)

TBL <- table(drybeans$Class)

print("Checking for imbalanced data...")
# CHECK FOR IMBALANCED DATA
if(any(TBL<(nrow(drybeans)*1/7*1/3) | any(TBL>(nrow(drybeans)*1/7*3)))) {
  imbalanced_classes=names(TBL)[TBL<(nrow(drybeans)*1/7*1/3) | TBL>(nrow(drybeans)*1/7*3)]
  print("Imbalanced Classes:")
  print(imbalanced_classes)
}

print("Basic descriptive statistics...")
newdata <- group_by(drybeans, Class)
newdata <- summarize(newdata, mean_perim = mean(MajorAxisLength, na.rm=TRUE))
newdata
newdata2 <- group_by(drybeans, Class)
newdata2 <- summarize(newdata2, mean_perim = mean(MinorAxisLength, na.rm=TRUE))
newdata2

print("Visualizing data...")# SOME VISUALIZATION (nonexhaustive)

ggplot(data = drybeans, aes(MajorAxisLength, AspectRation, color = Class)) + geom_point() 
+ scale_color_manual(values = c("BARBUNYA" = "red", "BOMBAY" = "blue", "CALI" = "yellow", 
                                "DERMASON"= "violetred3", "HOROZ" = "mediumorchid1", 
                                "SEKER" = "springgreen1", "SIRA" = "lightseagreen"))


f <- ggplot(drybeans, aes(Class, Perimeter)) + geom_boxplot()
f
f <- ggplot(drybeans, aes(Class, MajorAxisLength)) + geom_boxplot()
f
f <- ggplot(drybeans, aes(Class, MinorAxisLength)) + geom_boxplot()
f
f <- ggplot(drybeans, aes(Class, roundness)) + geom_boxplot()
f
f <- ggplot(drybeans, aes(Class, Compactness)) + geom_boxplot()
f
f <- ggplot(drybeans, aes(Class, Area)) + geom_boxplot()
f

print("Check for missing values...")
# CHECK FOR MISSING VALUES

allmisscols <- sapply(drybeans, function(x) all(is.na(x) | x == '' ))
allmisscols

print("Min-max normalization...")
# MIN-MAX NORMALIZATION
min_max_norm <- function(x) { (x - min(x)) / (max(x) - min(x)) }

db_norm <- as.data.frame(lapply(drybeans[1:16], min_max_norm))
head(db_norm)
db_norm$Class <- drybeans$Class
head(db_norm)

print("Get only numeric data for PCA...")
# GET ONLY NUMERIC DATA
numeric_vars <- c(
  "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength", "AspectRation", "Eccentricity",
  "ConvexArea", "EquivDiameter", "Extent", "Solidity", "roundness", 
  "Compactness", "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4"
)
db_numeric <- db_norm[, numeric_vars]
ncol(db_numeric)

print("Perform PCA...")
# PCA
res_pca <- prcomp(db_numeric, scale = TRUE)

print("See PCA rotation...")
# PCA ROTATION
res_pca$rotation[,1]
names(sort(abs(res_pca$rotation[,1]), decreasing=TRUE))

print("Check for outliers...")
# OUTLIERS
# Initialize a variable
outliers = c()
for(i in 1:16){
  data_mean = mean(db_numeric[,i])
  data_sd = sd(db_numeric[,i])
  low = data_mean - 3*data_sd
  up = data_mean + 3*data_sd
  index = which(db_numeric[,i] < low | db_numeric[,i] > up)
  outliers = c(outliers, index)
}

# Remove duplicates
outliers = unique(outliers)
print(paste("Number of outliers:",length(outliers)))

# HWK 2
library(naivebayes)
library(psych)
library(pROC)

# Starting the Naive Bayes classification
dt = sort(sample(nrow(drybeans), nrow(drybeans)*.7))
train<-drybeans[dt,]
test<-drybeans[-dt,]
model <- naive_bayes(train, train$Class, laplace = 1)

# PREDICT
p <- predict(model, train, type = 'prob')

# CONFUSION MATRIX
p1 <- predict(model, train)
(tab1 <- table(p1, train$Class))

mclass1 <- 1 - sum(diag(tab1)) / sum(tab1)
mclass1
# 0.0446

p2 <- predict(model, test)
(tab2 <- table(p2, test$Class))
p2
mclass2 <- 1 - sum(diag(tab2)) / sum(tab2)
mclass2

# ROC
mcroc <- multiclass.roc(test$Class, as.numeric(p2),levels=base::levels(as.factor(test$Class)),
               percent=FALSE)

# Must plot the ROC curves individually
for(i in 1:21){
  plot.roc(mcroc$rocs[[i]],legacy.axes=TRUE, add=TRUE)
  sapply(1:length(mcroc$rocs),function(i) lines.roc(mcroc$rocs[[i]],col=i))
}

# SPECIFICITY AND SENSITIVITY
library(mltest)
classifier_metrics <- ml_test(p2, test$Class, output.as.table = TRUE)
accuracy <- classifier_metrics$accuracy
precision <- classifier_metrics$precision
recall <- classifier_metrics$recall
specificity <- classifier_metrics$specificity

# SVM
library(e1071)
train_matrix <- as.matrix(train)
# remove the dependent variable from the training data set
tr <- subset(train, select=-Class)
# set the dependent variable as.factors
y <- as.factor(train$Class)
# create the SVM model where x = the data minus DV, and y = DV as.factors()
svm1 <- svm(tr, y, type="C-classification")
# the below won't work if the training data set has DV removed. but it won't work for other reasons too lol
# svm1 <- svm(Class~., data=train_matrix, type="C-classification")

summary(svm1)

# Predict on the test data
te <- subset(test, select=-Class)
pred <- predict(svm1, te)

# Check accuracy:
table(pred, y)

# ROC for SVM
library(pROC)
pred_numeric <- as.numeric(pred)
mc <- multiclass.roc(test$Class, pred_numeric)

# Must plot the ROC curves individually
for(i in 1:21){
  plot.roc(mc$rocs[[i]],legacy.axes=TRUE, add=TRUE)
  sapply(1:length(mc$rocs),function(i) lines.roc(mc$rocs[[i]],col=i))
}

# CONF MATRIX
conf_matrix <- table(pred, test$Class)
conf_matrix

# SPECIFICITY, SENSITIVITY, PRECISION, RECALL, ACCURACY -- SVM
library(mltest)
svm_metrics <- ml_test(pred, test$Class, output.as.table = FALSE)
accuracy <- svm_metrics$accuracy
precision <- svm_metrics$precision
recall <- svm_metrics$recall
specificity <- svm_metrics$specificity

# TREE
library(tree)
tr <- subset(train, select=-Class)
# <- tree(log10(perf) ~ syct+mmin+mmax+cach+chmin+chmax, cpus)
tree.model.4 <- tree(as.factor(Class)~ MajorAxisLength+ShapeFactor2+Perimeter+EquivDiameter+ConvexArea, drybeans)
plot(tree.model.4)
text(tree.model.4, pretty = 0, cex=0.5)
tree.pred <- predict(tree.model.4, test, type="class")

# Confusion Matrix
table(tree.pred, test$Class)

# ROC for Decision Tree
library(pROC)
pred_numeric <- as.numeric(tree.pred)
mc <- multiclass.roc(test$Class, pred_numeric)

# Must plot the ROC curves individually
for(i in 1:21){
  plot.roc(mc$rocs[[i]],legacy.axes=TRUE, add=TRUE)
  sapply(1:length(mc$rocs),function(i) lines.roc(mc$rocs[[i]],col=i))
}

# SPECIFICITY, SENSITIVITY, PRECISION, RECALL, ACCURACY -- DECISION TREE
library(mltest)
tree_metrics <- ml_test(tree.pred, test$Class, output.as.table = FALSE)
accuracy <- tree_metrics$accuracy
precision <- tree_metrics$precision
recall <- tree_metrics$recall
specificity <- tree_metrics$specificity


# K-NEAREST NEIGHBORS
library(caret)

# Split data into TEST and TRAIN
index <- createDataPartition(y = drybeans$Class,p = 0.7, list = FALSE)
train <- drybeans[index,]
test <- drybeans[-index,]

# Check distribution
prop.table(table(train$Class)) * 100
prop.table(table(test$Class)) * 100
prop.table(table(drybeans$Class)) * 100

# Preprocessing
trainX <- train[,names(train) != "Class"]
preProcValues <- preProcess(x = trainX, method = c("center", "scale"))
preProcValues

# Train
set.seed(1234)
ctrl <- trainControl(method="repeatedcv",repeats = 3) #,classProbs=TRUE,summaryFunction = twoClassSummary)
knnFit <- train(Class~., data = train, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)

#Output of kNN fit
knnFit

# Predict
knnPredict <- predict(knnFit, newdata = test)

# CONFUSION MATRIX
confusionMatrix(knnPredict, as.factor(test$Class))

# ROC for KNN
library(pROC)
knn_pred_numeric <- as.numeric(knnPredict)
mc_roc <- multiclass.roc(test$Class, knn_pred_numeric)

# Must plot the ROC curves individually
for(i in 1:21){
  plot.roc(mc_roc$rocs[[i]],legacy.axes=TRUE, add=TRUE)
  sapply(1:length(mc_roc$rocs),function(i) lines.roc(mc_roc$rocs[[i]],col=i))
}

# SPECIFICITY, SENSITIVITY, PRECISION, RECALL, ACCURACY -- DECISION TREE
library(mltest)
knn_metrics <- ml_test(knnPredict, test$Class, output.as.table = FALSE)
accuracy <- knn_metrics$accuracy
precision <- knn_metrics$precision
recall <- knn_metrics$recall
