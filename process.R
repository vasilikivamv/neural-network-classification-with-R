## Loading the relevant libraries 

library(tensorflow)
library(keras)
library(tidyverse)
library(Rmisc)
library(mvnormtest)
library(caret)
library(GGally)
library(ggplot2)
library(factoextra)
library(pROC)
library(corrplot)
library(MLmetrics)


## Load the PD data set, deleting column 28 and id

mydata <- read.delim ("D:/train_data.txt", header = FALSE, sep = ",")
colnames(mydata)[29] <- "parkinson"
mydata <- mydata[,-1]
mydata <- mydata[,-27]


# Check the distribution of the classes

d <- density(mydata$parkinson)
plot(d, type="n", main="parkinson")
polygon(d, col="dodgerblue", border="gray")



### Check for  Multivariate normality

x <- mydata
cm <- colMeans(x)
S <- cov(x)
d <- apply(x, 1, function(x) t(x - cm) %*% solve(S) %*% (x - cm))

# Chi-Square plot:
plot(qchisq((1:nrow(x) - 1/2) / nrow(x), df = ncol(x)), 
     sort(d),
     xlab = expression(paste(chi[22]^2, 
                             " Quantile")), 
     ylab = "Ordered distances", col="dodgerblue")
abline(a = 0, b = 1, col="red")

#From the above figure, we can see that the data are not normally distributed.
#There are many outliers.

#### Check correlations between the variables with heatmap

corrplot(cor(mydata), type="full", method ="color",
         title = "Parkinson correlation plot", mar=c(0,0,1,0), tl.cex= 0.8, outline= T, tl.col="indianred4")


### Conduct the PCA
parkinson_pca <- prcomp(mydata, scale = T)

### Get the eigenvectors 
parkinson_pca$rotation

summary(parkinson_pca)

pcvars <- parkinson_pca$sdev^2             # Eigenvalues

# variance explained
var.explained <- cumsum(pcvars)/sum(pcvars)

# plot of Proportion of variance explained
plot(seq_along(pcvars), pcvars/sum(pcvars), type="o", ylim=c(0,1), xlab="k", ylab="Proportion of variance explained")
points(seq_along(pcvars), var.explained, lty=2, type="o")
legend("right", lty=c(1,2), legend=c("Individual","Cumulative"))
abline(h=c(1/ncol(mydata), 0.9), lty=3)


# c=90% of variance explained
min(which(var.explained>=0.9))    ## with 8 PC we get 90% explanation of the data set


fviz_pca_biplot(parkinson_pca, col.ind = mydata$parkinson,
                palette = "lancet", geom = "point")

# Dimension (Dim.) 1 and 2 retained about 55.6% (16.6% + 39%) of the total information contained in the data set.
# Variables that are positively correlated are on the same side of the plots.
#Variables that are negatively correlated are on the opposite side of the plots


## Splitting the data set to training and testing sets

set.seed(123)                       # set seed for reproducibility
N <- nrow(mydata)
train_id <- sample(1:N, N*0.6, replace = F)
train.set <- mydata[train_id,]                  ## training set
test.set <- mydata[-train_id,]                  ## test set


# Scaling the data for the neural network and do one-hot-encoding

X_train <- scale(train.set[, c(1:26)])                             # input for training
y_train <- to_categorical(train.set$parkinson, num_classes = 2)

X_test <- scale(test.set[, c(1:26)])
y_test <- to_categorical(test.set$parkinson, num_classes = 2)  # what we want to predict

