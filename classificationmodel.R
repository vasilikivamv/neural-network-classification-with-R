source("process.R")   # source the data pre-processing Rscript

                                  #### SGD investigation ####

                                   


                                 #### Number of Neurons ####

# initialize matrices to store the results
train.loss <- matrix(ncol = 2, nrow = 50)
train.acc <- matrix(ncol = 2, nrow =50)
test.loss <- matrix(ncol = 2, nrow = 50)
test.acc <- matrix(ncol = 2, nrow = 50)
k = 1
loss.tr <- matrix(ncol = 50, nrow = 100)
loss.tst <- matrix(ncol = 50, nrow = 100)
acc.tr <- matrix(ncol = 50, nrow = 100)
acc.tst <- matrix(ncol = 50, nrow = 100)



for (neurons in c(5,10,15,20,25)) {
  for (i in 1:10) {
    
    base_model <- keras_model_sequential() 
    base_model %>% 
      layer_dense(units = neurons, activation = 'relu', input_shape = ncol(X_train)) %>%
      layer_dense(units = 2, activation = 'softmax')
    
    
    
    base_model %>% compile(
      loss = 'binary_crossentropy',
      optimizer = optimizer_sgd(
        lr = 0.01,
        momentum = 0.9,
        decay = 1e-4,
        nesterov = FALSE,
        clipnorm = 1,
        clipvalue =0.5),
      metrics = c('accuracy'))
    
    
    history <- base_model %>% fit(
      X_train, y_train, 
      epochs = 100, batch_size = 50, 
      validation_split = 0.3, shuffle=TRUE)
    
    loss.tr[,k] <- history$metrics$loss
    loss.tst[,k] <- history$metrics$val_loss
    acc.tr[,k] <- history$metrics$accuracy
    acc.tst[,k] <- history$metrics$val_accuracy
    
    
    results.train <- base_model %>% evaluate(X_train, y_train)
    train.loss[k,1] <- as.integer(neurons)
    train.loss[k,2] <- results.train$loss
    
    train.acc[k,1] <-  neurons
    train.acc[k,2] <- results.train$accuracy
    
    results.test <- base_model %>% evaluate(X_test, y_test)
    test.loss[k,1] <- neurons
    test.loss[k,2] <- results.test$loss
    
    test.acc[k,1] <-  neurons
    test.acc[k,2] <- results.test$accuracy
    
    k = k+1
    
  }
  
}


## Calculates mean and 95% CI
SGD.trainLOSS <- c(CI(train.loss[1:10,2]),CI(train.loss[11:20,2]),
                   CI(train.loss[21:30,2]),CI(train.loss[31:40,2]),CI(train.loss[41:50,2]))
SDG.testLOSS <- c(CI(test.loss[1:10,2]),CI(test.loss[11:20,2]),
                  CI(test.loss[21:30,2]),CI(test.loss[31:40,2]),CI(test.loss[41:50,2]))
SGD.trainACC <- c(CI(train.acc[1:10,2]),CI(train.acc[11:20,2]),
                  CI(train.acc[21:30,2]),CI(train.acc[31:40,2]),CI(train.acc[41:50,2]))
SGD.testACC <- c(CI(test.acc[1:10,2]),CI(test.acc[11:20,2]),
                 CI(test.acc[21:30,2]),CI(test.acc[31:40,2]),CI(test.acc[41:50,2]))


# Training loss Plot

ResultLoss_mean_tr<- matrix(ncol = 5, nrow = 100)
l=10 #number of experiments
h=1 
for (i in 1:100){
  
  ResultLoss_mean_tr[i,h] = mean(loss.tr[i,1:10])
  ResultLoss_mean_tr[i,h+1] = mean(loss.tr[i,11:20])
  ResultLoss_mean_tr[i,h+2] = mean(loss.tr[i,21:30])
  ResultLoss_mean_tr[i,h+3] = mean(loss.tr[i,31:40])
  ResultLoss_mean_tr[i,h+4] = mean(loss.tr[i,41:50])
  
}
plot(1:100,ResultLoss_mean_tr[,1], type = "l", col="dodgerblue", lwd=2,ylim=c(0.2,1), ylab = "Loss", xlab = "Epochs",
     main = "Average Training Loss for SGD")
lines(ResultLoss_mean_tr[,2],lwd=2, col="purple")
lines(ResultLoss_mean_tr[,3], lwd=2,col="indianred")
lines(ResultLoss_mean_tr[,4], lwd=2,col="green")
lines(ResultLoss_mean_tr[,5], lwd=2,col="orange")
legend("topright", legend=c("5 neurons", "10 neurons","15 neurons", "20 neurons", "25 neurons"),
       col=c("dodgerblue", "purple","indianred","green","orange"), lwd = 2, cex=0.8, box.lwd=2)
grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted",
     lwd = par("lwd"), equilogs = TRUE)


## Test loss Plot
ResultLoss_mean_tst<- matrix(ncol = 5, nrow = 100)
l=10 #number of experiments
h=1 
for (i in 1:100){
  
  ResultLoss_mean_tst[i,h] = mean(loss.tst[i,1:10])
  ResultLoss_mean_tst[i,h+1] = mean(loss.tst[i,11:20])
  ResultLoss_mean_tst[i,h+2] = mean(loss.tst[i,21:30])
  ResultLoss_mean_tst[i,h+3] = mean(loss.tst[i,31:40])
  ResultLoss_mean_tst[i,h+4] = mean(loss.tst[i,41:50])
  
}

plot(1:100,ResultLoss_mean_tst[,1], type = "l", col="dodgerblue", lwd=2,ylim=c(0.6,1), ylab = "Loss", xlab = "Epochs",
     main = "Average Test Loss for SGD")
lines(ResultLoss_mean_tst[,2],lwd=2, col="purple")
lines(ResultLoss_mean_tst[,3], lwd=2,col="indianred")
lines(ResultLoss_mean_tst[,4], lwd=2,col="green")
lines(ResultLoss_mean_tst[,5], lwd=2,col="orange")
legend("topright", legend=c("5 neurons", "10 neurons","15 neurons", "20 neurons", "25 neurons"),
       col=c("dodgerblue", "purple","indianred","green","orange"), lwd = 2, cex=0.8, box.lwd=2)
grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted",
     lwd = par("lwd"), equilogs = TRUE)



# Test accuracy Plot
ResultAcc_mean_tst<- matrix(ncol = 5, nrow = 100)
l=10 #number of experiments
h=1 
for (i in 1:100){
  
  ResultAcc_mean_tst[i,h] = mean(acc.tst[i,1:10])
  ResultAcc_mean_tst[i,h+1] = mean(acc.tst[i,11:20])
  ResultAcc_mean_tst[i,h+2] = mean(acc.tst[i,21:30])
  ResultAcc_mean_tst[i,h+3] = mean(acc.tst[i,31:40])
  ResultAcc_mean_tst[i,h+4] = mean(acc.tst[i,41:50])
  
}

plot(1:100,ResultAcc_mean_tst[,1], type = "l", col="dodgerblue", lwd=2,ylim=c(0.4,0.8), ylab = "Accuracy", xlab = "Epochs",
     main = "Average Test Accuracy for SGD")
lines(ResultAcc_mean_tst[,2],lwd=2, col="purple")
lines(ResultAcc_mean_tst[,3], lwd=2,col="indianred")
lines(ResultAcc_mean_tst[,4], lwd=2,col="green")
lines(ResultAcc_mean_tst[,5], lwd=2,col="orange")
legend("bottomright", legend=c("5 neurons", "10 neurons","15 neurons", "20 neurons", "25 neurons"),
       col=c("dodgerblue", "purple","indianred","green","orange"), lwd = 2, cex=0.8, box.lwd=2)
grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted",
     lwd = par("lwd"), equilogs = TRUE)



# Train accuracy Plot
ResultAcc_mean_tr<- matrix(ncol = 5, nrow = 100)
l=10 #number of experiments
h=1 
for (i in 1:100){
  
  ResultAcc_mean_tr[i,h] = mean(acc.tr[i,1:10])
  ResultAcc_mean_tr[i,h+1] = mean(acc.tr[i,11:20])
  ResultAcc_mean_tr[i,h+2] = mean(acc.tr[i,21:30])
  ResultAcc_mean_tr[i,h+3] = mean(acc.tr[i,31:40])
  ResultAcc_mean_tr[i,h+4] = mean(acc.tr[i,41:50])
  
}

plot(1:100,ResultAcc_mean_tr[,1], type = "l", col="dodgerblue", lwd=2,ylim=c(0.4,1), ylab = "Accuracy", xlab = "Epochs",
     main = "Average Train Accuracy for SGD")
lines(ResultAcc_mean_tr[,2],lwd=2, col="purple")
lines(ResultAcc_mean_tr[,3], lwd=2,col="indianred")
lines(ResultAcc_mean_tr[,4], lwd=2,col="green")
lines(ResultAcc_mean_tr[,5], lwd=2,col="orange")
legend("bottomright", legend=c("5 neurons", "10 neurons","15 neurons", "20 neurons", "25 neurons"),
       col=c("dodgerblue", "purple","indianred","green","orange"), lwd = 2, cex=0.8, box.lwd=2)
grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted",
     lwd = par("lwd"), equilogs = TRUE)





                                  #### Learning rates ####

# initialize matrices to store the results

train.loss <- matrix(ncol = 2, nrow = 50)
train.acc <- matrix(ncol = 2, nrow =50)
test.loss <- matrix(ncol = 2, nrow = 50)
test.acc <- matrix(ncol = 2, nrow = 50)
k = 1
loss.tr <- matrix(ncol = 50, nrow = 100)
loss.tst <- matrix(ncol = 50, nrow = 100)
acc.tr <- matrix(ncol = 50, nrow = 100)
acc.tst <- matrix(ncol = 50, nrow = 100)


for (learning.rate in c(0.1,0.2,0.3,0.5,1)) {
  for (i in 1:10) {
    
    model.lr <- keras_model_sequential() 
    model.lr %>% 
      layer_dense(units = 25, activation = 'relu', input_shape = ncol(X_train)) %>% 
      layer_dense(units = 2, activation = 'softmax')
    
    model.lr %>% compile(
      loss = 'binary_crossentropy',
      optimizer = optimizer_sgd(
        lr = learning.rate,
        momentum = 0.9,
        decay = 1e-4,
        nesterov = FALSE,
        clipnorm = 1,
        clipvalue = 0.5),
      metrics = c('accuracy'))
    
    
    history.lr<- model.lr %>% fit(
      X_train, y_train, 
      epochs = 100, batch_size = 50, 
      validation_split = 0.3, shuffle=TRUE)
    
    loss.tr[,k] <- history.lr$metrics$loss
    loss.tst[,k] <- history.lr$metrics$val_loss
    acc.tr[,k] <- history.lr$metrics$accuracy
    acc.tst[,k] <- history.lr$metrics$val_accuracy
    
    
    
    results.train <- model.lr %>% evaluate(X_train, y_train)
    train.loss[k,1] <- learning.rate
    train.loss[k,2] <- results.train$loss
    
    train.acc[k,1] <-  learning.rate
    train.acc[k,2] <- results.train$accuracy
    
    results.test <- model.lr %>% evaluate(X_test, y_test)
    test.loss[k,1] <- learning.rate
    test.loss[k,2] <- results.test$loss
    
    test.acc[k,1] <- learning.rate
    test.acc[k,2] <- results.test$accuracy
    
    k = k+1
    
  }
  
}


# Calculates the mean and 95% CI
LRSGD.trainLOSS<- c(CI(train.loss[1:10,2]),CI(train.loss[11:20,2]),
                    CI(train.loss[21:30,2]),CI(train.loss[31:40,2]),CI(train.loss[41:50,2]))
LRSGD.testLOSS<- c(CI(test.loss[1:10,2]),CI(test.loss[11:20,2]),
                   CI(test.loss[21:30,2]),CI(test.loss[31:40,2]),CI(test.loss[41:50,2]))
LRSGD.trainACC <- c(CI(train.acc[1:10,2]),CI(train.acc[11:20,2]),
                    CI(train.acc[21:30,2]),CI(train.acc[31:40,2]),CI(train.acc[41:50,2]))
LRSGD.testACC <- c(CI(test.acc[1:10,2]),CI(test.acc[11:20,2]),
                   CI(test.acc[21:30,2]),CI(test.acc[31:40,2]),CI(test.acc[41:50,2]))



# Training loss Plot
ResultLoss_mean_tr<- matrix(ncol = 5, nrow = 100)
l=10 #number of experiments
h=1 
for (i in 1:100){
  
  ResultLoss_mean_tr[i,h] = mean(loss.tr[i,1:10])
  ResultLoss_mean_tr[i,h+1] = mean(loss.tr[i,11:20])
  ResultLoss_mean_tr[i,h+2] = mean(loss.tr[i,21:30])
  ResultLoss_mean_tr[i,h+3] = mean(loss.tr[i,31:40])
  ResultLoss_mean_tr[i,h+4] = mean(loss.tr[i,41:50])
  
}
plot(1:100,ResultLoss_mean_tr[,1], type = "l", col="dodgerblue", lwd=2,ylim=c(0,3), ylab = "Loss", xlab = "Epochs",
     main = "Average Training Loss for SGD")
lines(ResultLoss_mean_tr[,2],lwd=2, col="purple")
lines(ResultLoss_mean_tr[,3], lwd=2,col="indianred")
lines(ResultLoss_mean_tr[,4], lwd=2,col="green")
lines(ResultLoss_mean_tr[,5], lwd=2,col="orange")
legend("topright", legend=c("lr: 0.1", "lr: 0.2","lr: 0.3", "lr: 0.5", "lr: 1"),
       col=c("dodgerblue", "purple","indianred","green","orange"), lwd = 2, cex=0.8, box.lwd=2)
grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted",
     lwd = par("lwd"), equilogs = TRUE)


## Test loss Plot
ResultLoss_mean_tst<- matrix(ncol = 5, nrow = 100)
l=10 #number of experiments
h=1 
for (i in 1:100){
  
  ResultLoss_mean_tst[i,h] = mean(loss.tst[i,1:10])
  ResultLoss_mean_tst[i,h+1] = mean(loss.tst[i,11:20])
  ResultLoss_mean_tst[i,h+2] = mean(loss.tst[i,21:30])
  ResultLoss_mean_tst[i,h+3] = mean(loss.tst[i,31:40])
  ResultLoss_mean_tst[i,h+4] = mean(loss.tst[i,41:50])
  
}

plot(1:100,ResultLoss_mean_tst[,1], type = "l", col="dodgerblue", lwd=2,ylim=c(0,4), ylab = "Loss", xlab = "Epochs",
     main = "Average Test Loss for SGD")
lines(ResultLoss_mean_tst[,2],lwd=2, col="purple")
lines(ResultLoss_mean_tst[,3], lwd=2,col="indianred")
lines(ResultLoss_mean_tst[,4], lwd=2,col="green")
lines(ResultLoss_mean_tst[,5], lwd=2,col="orange")
legend("topright", legend=c("lr: 0.1", "lr: 0.2","lr: 0.3", "lr: 0.5", "lr: 1"),
       col=c("dodgerblue", "purple","indianred","green","orange"), lwd = 2, cex=0.8, box.lwd=2)
grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted",
     lwd = par("lwd"), equilogs = TRUE)



# Test accuracy Plot
ResultAcc_mean_tst<- matrix(ncol = 5, nrow = 100)
l=10 #number of experiments
h=1 
for (i in 1:100){
  
  ResultAcc_mean_tst[i,h] = mean(acc.tst[i,1:10])
  ResultAcc_mean_tst[i,h+1] = mean(acc.tst[i,11:20])
  ResultAcc_mean_tst[i,h+2] = mean(acc.tst[i,21:30])
  ResultAcc_mean_tst[i,h+3] = mean(acc.tst[i,31:40])
  ResultAcc_mean_tst[i,h+4] = mean(acc.tst[i,41:50])
  
}

plot(1:100,ResultAcc_mean_tst[,1], type = "l", col="dodgerblue", lwd=2,ylim=c(0.4,1), ylab = "Accuracy", xlab = "Epochs",
     main = "Average Test Accuracy for SGD")
lines(ResultAcc_mean_tst[,2],lwd=2, col="purple")
lines(ResultAcc_mean_tst[,3], lwd=2,col="indianred")
lines(ResultAcc_mean_tst[,4], lwd=2,col="green")
lines(ResultAcc_mean_tst[,5], lwd=2,col="orange")
legend("topright", legend=c("lr: 0.1", "lr: 0.2","lr: 0.3", "lr: 0.5", "lr: 1"),
       col=c("dodgerblue", "purple","indianred","green","orange"), lwd = 2, cex=0.8, box.lwd=2)
grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted",
     lwd = par("lwd"), equilogs = TRUE)



# Train accuracy Plot
ResultAcc_mean_tr<- matrix(ncol = 5, nrow = 100)
l=10 #number of experiments
h=1 
for (i in 1:100){
  
  ResultAcc_mean_tr[i,h] = mean(acc.tr[i,1:10])
  ResultAcc_mean_tr[i,h+1] = mean(acc.tr[i,11:20])
  ResultAcc_mean_tr[i,h+2] = mean(acc.tr[i,21:30])
  ResultAcc_mean_tr[i,h+3] = mean(acc.tr[i,31:40])
  ResultAcc_mean_tr[i,h+4] = mean(acc.tr[i,41:50])
  
}

plot(1:100,ResultAcc_mean_tr[,1], type = "l", col="dodgerblue", lwd=2,ylim=c(0.2,1), ylab = "Accuracy", xlab = "Epochs",
     main = "Average Train Accuracy for SGD")
lines(ResultAcc_mean_tr[,2],lwd=2, col="purple")
lines(ResultAcc_mean_tr[,3], lwd=2,col="indianred")
lines(ResultAcc_mean_tr[,4], lwd=2,col="green")
lines(ResultAcc_mean_tr[,5], lwd=2,col="orange")
legend("bottomright", legend=c("lr: 0.1", "lr: 0.2","lr: 0.3", "lr: 0.5", "lr: 1"),
       col=c("dodgerblue", "purple","indianred","green","orange"), lwd = 2, cex=0.8, box.lwd=2)
grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted",
     lwd = par("lwd"), equilogs = TRUE)




                                  #### Momentum Rates ####

# Initialize matrices to store outputs
train.loss <- matrix(ncol = 2, nrow = 60)
train.acc <- matrix(ncol = 2, nrow =60)
test.loss <- matrix(ncol = 2, nrow = 60)
test.acc <- matrix(ncol = 2, nrow = 60)
k = 1
loss.tr <- matrix(ncol = 60, nrow = 100)
loss.tst <- matrix(ncol = 60, nrow = 100)
acc.tr <- matrix(ncol = 60, nrow = 100)
acc.tst <- matrix(ncol = 60, nrow = 100)


for (momentum.rate in c(0.01,0.02,0.04,0.07,0.1,0.9)) {
  for (i in 1:10) {
    
    model.mr <- keras_model_sequential() 
    model.mr %>% 
      layer_dense(units = 25, activation = 'relu', input_shape = ncol(X_train)) %>% 
      layer_dense(units = 2, activation = 'softmax')
    
    model.mr %>% compile(
      loss = 'binary_crossentropy',
      optimizer = optimizer_sgd(
        lr = 0.1,
        momentum = momentum.rate,
        decay = 1e-4,
        nesterov = FALSE,
        clipnorm = 1,
        clipvalue = 0.5),
      metrics = c('accuracy'))
    
    
    history.mr <- model.mr %>% fit(
      X_train, y_train, 
      epochs = 100, batch_size = 50, 
      validation_split = 0.3, shuffle=TRUE)
    
    loss.tr[,k] <- history.mr$metrics$loss
    loss.tst[,k] <- history.mr$metrics$val_loss
    acc.tr[,k] <- history.mr$metrics$accuracy
    acc.tst[,k] <- history.mr$metrics$val_accuracy
    
    results.train <- model.mr %>% evaluate(X_train, y_train)
    train.loss[k,1] <- momentum.rate
    train.loss[k,2] <- results.train$loss
    
    train.acc[k,1] <-  momentum.rate
    train.acc[k,2] <- results.train$accuracy
    
    results.test <- model.mr %>% evaluate(X_test, y_test)
    test.loss[k,1] <- momentum.rate
    test.loss[k,2] <- results.test$loss
    
    test.acc[k,1] <-  momentum.rate
    test.acc[k,2] <- results.test$accuracy
    
    k = k+1
    
  }
  
}

# Calculates meand and 95% CI
MRSGD.trainLOSS<- c(CI(train.loss[1:10,2]),CI(train.loss[11:20,2]),
                    CI(train.loss[21:30,2]),CI(train.loss[31:40,2]),CI(train.loss[41:50,2]))
MRSGD.testLOSS<- c(CI(test.loss[1:10,2]),CI(test.loss[11:20,2]),
                   CI(test.loss[21:30,2]),CI(test.loss[31:40,2]),CI(test.loss[41:50,2]))
MRSGD.trainACC <- c(CI(train.acc[1:10,2]),CI(train.acc[11:20,2]),
                    CI(train.acc[21:30,2]),CI(train.acc[31:40,2]),CI(train.acc[41:50,2]))
MRSGD.testACC <- c(CI(test.acc[1:10,2]),CI(test.acc[11:20,2]),
                   CI(test.acc[21:30,2]),CI(test.acc[31:40,2]),CI(test.acc[41:50,2]))


# Training loss Plot
ResultLoss_mean_tr<- matrix(ncol = 6, nrow = 100)
l=10 #number of experiments
h=1 
for (i in 1:100){
  
  ResultLoss_mean_tr[i,h] = mean(loss.tr[i,1:10])
  ResultLoss_mean_tr[i,h+1] = mean(loss.tr[i,11:20])
  ResultLoss_mean_tr[i,h+2] = mean(loss.tr[i,21:30])
  ResultLoss_mean_tr[i,h+3] = mean(loss.tr[i,31:40])
  ResultLoss_mean_tr[i,h+4] = mean(loss.tr[i,41:50])
  ResultLoss_mean_tr[i,h+5] = mean(loss.tr[i,51:60])
  
}
plot(1:100,ResultLoss_mean_tr[,1], type = "l", col="dodgerblue", lwd=2,ylim=c(0.2,1), ylab = "Loss", xlab = "Epochs",
     main = "Average Training Loss for SGD")
lines(ResultLoss_mean_tr[,2],lwd=2, col="purple")
lines(ResultLoss_mean_tr[,3], lwd=2,col="indianred")
lines(ResultLoss_mean_tr[,4], lwd=2,col="green")
lines(ResultLoss_mean_tr[,5], lwd=2,col="orange")
lines(ResultLoss_mean_tr[,6], lwd=2,col="black")
legend("topright", legend=c("mr: 0.01", "mr: 0.02","mr: 0.04", "mr: 0.07", "mr: 0.1","mr: 0.9"),
       col=c("dodgerblue", "purple","indianred","green","orange","black"), lwd = 2, cex=0.8, box.lwd=2)
grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted",
     lwd = par("lwd"), equilogs = TRUE)


## Test loss Plot
ResultLoss_mean_tst<- matrix(ncol = 6, nrow = 100)
l=10 #number of experiments
h=1 
for (i in 1:100){
  
  ResultLoss_mean_tst[i,h] = mean(loss.tst[i,1:10])
  ResultLoss_mean_tst[i,h+1] = mean(loss.tst[i,11:20])
  ResultLoss_mean_tst[i,h+2] = mean(loss.tst[i,21:30])
  ResultLoss_mean_tst[i,h+3] = mean(loss.tst[i,31:40])
  ResultLoss_mean_tst[i,h+4] = mean(loss.tst[i,41:50])
  ResultLoss_mean_tst[i,h+5] = mean(loss.tst[i,51:60])
  
}

plot(1:100,ResultLoss_mean_tst[,1], type = "l", col="dodgerblue", lwd=2,ylim=c(0.4,2), ylab = "Loss", xlab = "Epochs",
     main = "Average Test Loss for SGD")
lines(ResultLoss_mean_tst[,2],lwd=2, col="purple")
lines(ResultLoss_mean_tst[,3], lwd=2,col="indianred")
lines(ResultLoss_mean_tst[,4], lwd=2,col="green")
lines(ResultLoss_mean_tst[,5], lwd=2,col="orange")
lines(ResultLoss_mean_tst[,6], lwd=2,col="black")
legend("topleft", legend=c("mr: 0.01", "mr: 0.02","mr: 0.04", "mr: 0.07", "mr: 0.1","mr: 0.9"),
       col=c("dodgerblue", "purple","indianred","green","orange","black"), lwd = 2, cex=0.8, box.lwd=2)
grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted",
     lwd = par("lwd"), equilogs = TRUE)



# Test accuracy Plot
ResultAcc_mean_tst<- matrix(ncol = 6, nrow = 100)
l=10 #number of experiments
h=1 
for (i in 1:100){
  
  ResultAcc_mean_tst[i,h] = mean(acc.tst[i,1:10])
  ResultAcc_mean_tst[i,h+1] = mean(acc.tst[i,11:20])
  ResultAcc_mean_tst[i,h+2] = mean(acc.tst[i,21:30])
  ResultAcc_mean_tst[i,h+3] = mean(acc.tst[i,31:40])
  ResultAcc_mean_tst[i,h+4] = mean(acc.tst[i,41:50])
  ResultAcc_mean_tst[i,h+5] = mean(loss.tst[i,51:60])
  
}

plot(1:100,ResultAcc_mean_tst[,1], type = "l", col="dodgerblue", lwd=2,ylim=c(0.4,2), ylab = "Accuracy", xlab = "Epochs",
     main = "Average Test Accuracy for SGD")
lines(ResultAcc_mean_tst[,2],lwd=2, col="purple")
lines(ResultAcc_mean_tst[,3], lwd=2,col="indianred")
lines(ResultAcc_mean_tst[,4], lwd=2,col="green")
lines(ResultAcc_mean_tst[,5], lwd=2,col="orange")
lines(ResultAcc_mean_tst[,6], lwd=2,col="black")
legend("topleft", legend=c("mr: 0.01", "mr: 0.02","mr: 0.04", "mr: 0.07", "mr: 0.1","mr: 0.9"),
       col=c("dodgerblue", "purple","indianred","green","orange","black"), lwd = 2, cex=0.8, box.lwd=2)
grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted",
     lwd = par("lwd"), equilogs = TRUE)



# Train accuracy Plot
ResultAcc_mean_tr<- matrix(ncol = 5, nrow = 100)
l=10 #number of experiments
h=1 
for (i in 1:100){
  
  ResultAcc_mean_tr[i,h] = mean(acc.tr[i,1:10])
  ResultAcc_mean_tr[i,h+1] = mean(acc.tr[i,11:20])
  ResultAcc_mean_tr[i,h+2] = mean(acc.tr[i,21:30])
  ResultAcc_mean_tr[i,h+3] = mean(acc.tr[i,31:40])
  ResultAcc_mean_tr[i,h+4] = mean(acc.tr[i,41:50])
  ResultAcc_mean_tr[i,h+5] = mean(acc.tr[i,51:60])
  
}

plot(1:100,ResultAcc_mean_tr[,1], type = "l", col="dodgerblue", lwd=2,ylim=c(0.4,2), ylab = "Accuracy", xlab = "Epochs",
     main = "Average Train Accuracy for SGD")
lines(ResultAcc_mean_tr[,2],lwd=2, col="purple")
lines(ResultAcc_mean_tr[,3], lwd=2,col="indianred")
lines(ResultAcc_mean_tr[,4], lwd=2,col="green")
lines(ResultAcc_mean_tr[,5], lwd=2,col="orange")
lines(ResultAcc_mean_tr[,5], lwd=2,col="black")
legend("bottomright", legend=c("mr: 0.01", "mr: 0.02","mr: 0.04", "mr: 0.07", "mr: 0.1","mr: 0.9"),
       col=c("dodgerblue", "purple","indianred","green","orange","black"), lwd = 2, cex=0.8, box.lwd=2)
grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted",
     lwd = par("lwd"), equilogs = TRUE)



                                  #### ADAM ####



                              #### Number of Neurons ####

# Initialize matrices to store the results
train.loss <- matrix(ncol = 2, nrow = 50)
train.acc <- matrix(ncol = 2, nrow =50)
test.loss <- matrix(ncol = 2, nrow = 50)
test.acc <- matrix(ncol = 2, nrow = 50)
k = 1
loss.tr <- matrix(ncol = 50, nrow = 100)
loss.tst <- matrix(ncol = 50, nrow = 100)
acc.tr <- matrix(ncol = 50, nrow = 100)
acc.tst <- matrix(ncol = 50, nrow = 100)


# Neural Network Model
for (neurons in c(5,10,15,20,25)) {
  
  for (i in 1:10) {
    
    base_model <- keras_model_sequential() 
    base_model %>% 
      layer_dense(units = neurons, activation = 'relu', input_shape = ncol(X_train)) %>% 
      layer_dense(units = 2, activation = 'softmax')
    
    
    base_model %>% compile(
      loss = 'binary_crossentropy',
      optimizer = optimizer_adam(
        lr = 0.1,
        beta_1 = 0.9,
        beta_2 = 0.999,
        decay = 1e-4,
        amsgrad = FALSE,
        clipnorm = 1,
        clipvalue = 0.5),
      metrics = c('accuracy'))
    
    
    history <- base_model %>% fit(
      X_train, y_train, 
      epochs = 100, batch_size = 50, 
      validation_split = 0.3, shuffle=TRUE)
    
    loss.tr[,k] <- history$metrics$loss
    loss.tst[,k] <- history$metrics$val_loss
    acc.tr[,k] <- history$metrics$accuracy
    acc.tst[,k] <- history$metrics$val_accuracy
    
    
    
    results.train <- base_model %>% evaluate(X_train, y_train)
    train.loss[k,1] <- as.integer(neurons)
    train.loss[k,2] <- results.train$loss
    
    train.acc[k,1] <-  neurons
    train.acc[k,2] <- results.train$accuracy
    
    results.test <- base_model %>% evaluate(X_test, y_test)
    test.loss[k,1] <- neurons
    test.loss[k,2] <- results.test$loss
    
    test.acc[k,1] <-  neurons
    test.acc[k,2] <- results.test$accuracy
    
    k = k+1
    
  }
  
}

# Mean and 95%CI
res.trainLOSS <- c(CI(train.loss[1:10,2]),CI(train.loss[11:20,2]),
                   CI(train.loss[21:30,2]),CI(train.loss[31:40,2]),CI(train.loss[41:50,2]))
res.testLOSS <- c(CI(test.loss[1:10,2]),CI(test.loss[11:20,2]),
                  CI(test.loss[21:30,2]),CI(test.loss[31:40,2]),CI(test.loss[41:50,2]))
res.trainACC <- c(CI(train.acc[1:10,2]),CI(train.acc[11:20,2]),
                  CI(train.acc[21:30,2]),CI(train.acc[31:40,2]),CI(train.acc[41:50,2]))
res.testACC <- c(CI(test.acc[1:10,2]),CI(test.acc[11:20,2]),
                 CI(test.acc[21:30,2]),CI(test.acc[31:40,2]),CI(test.acc[41:50,2]))

# Plots


# Training loss
ResultLoss_mean_tr<- matrix(ncol = 5, nrow = 100)
l=10 #number of experiments
h=1 
for (i in 1:100){
  
  ResultLoss_mean_tr[i,h] = mean(loss.tr[i,1:10])
  ResultLoss_mean_tr[i,h+1] = mean(loss.tr[i,11:20])
  ResultLoss_mean_tr[i,h+2] = mean(loss.tr[i,21:30])
  ResultLoss_mean_tr[i,h+3] = mean(loss.tr[i,31:40])
  ResultLoss_mean_tr[i,h+4] = mean(loss.tr[i,41:50])
  
}
plot(1:100,ResultLoss_mean_tr[,1], type = "l", col="dodgerblue", lwd=2,ylim=c(0.2,1), ylab = "Loss", xlab = "Epochs",
     main = "Average Training Loss for ADAM")
lines(ResultLoss_mean_tr[,2],lwd=2, col="purple")
lines(ResultLoss_mean_tr[,3], lwd=2,col="indianred")
lines(ResultLoss_mean_tr[,4], lwd=2,col="green")
lines(ResultLoss_mean_tr[,5], lwd=2,col="orange")
legend("topright", legend=c("5 neurons", "10 neurons","15 neurons", "20 neurons", "25 neurons"),
       col=c("dodgerblue", "purple","indianred","green","orange"), lwd = 2, cex=0.8, box.lwd=2)
grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted",
     lwd = par("lwd"), equilogs = TRUE)


## Test loss
ResultLoss_mean_tst<- matrix(ncol = 5, nrow = 100)
l=10 #number of experiments
h=1 
for (i in 1:100){
  
  ResultLoss_mean_tst[i,h] = mean(loss.tst[i,1:10])
  ResultLoss_mean_tst[i,h+1] = mean(loss.tst[i,11:20])
  ResultLoss_mean_tst[i,h+2] = mean(loss.tst[i,21:30])
  ResultLoss_mean_tst[i,h+3] = mean(loss.tst[i,31:40])
  ResultLoss_mean_tst[i,h+4] = mean(loss.tst[i,41:50])
  
}

plot(1:100,ResultLoss_mean_tst[,1], type = "l", col="dodgerblue", lwd=2,ylim=c(0.4,1), ylab = "Loss", xlab = "Epochs",
     main = "Average Test Loss for ADAM")
lines(ResultLoss_mean_tst[,2],lwd=2, col="purple")
lines(ResultLoss_mean_tst[,3], lwd=2,col="indianred")
lines(ResultLoss_mean_tst[,4], lwd=2,col="green")
lines(ResultLoss_mean_tst[,5], lwd=2,col="orange")
legend("bottomright", legend=c("5 neurons", "10 neurons","15 neurons", "20 neurons", "25 neurons"),
       col=c("dodgerblue", "purple","indianred","green","orange"), lwd = 2, cex=0.8, box.lwd=2)
grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted",
     lwd = par("lwd"), equilogs = TRUE)



# Test accuracy
ResultAcc_mean_tst<- matrix(ncol = 5, nrow = 100)
l=10 #number of experiments
h=1 
for (i in 1:100){
  
  ResultAcc_mean_tst[i,h] = mean(acc.tst[i,1:10])
  ResultAcc_mean_tst[i,h+1] = mean(acc.tst[i,11:20])
  ResultAcc_mean_tst[i,h+2] = mean(acc.tst[i,21:30])
  ResultAcc_mean_tst[i,h+3] = mean(acc.tst[i,31:40])
  ResultAcc_mean_tst[i,h+4] = mean(acc.tst[i,41:50])
  
}

plot(1:100,ResultAcc_mean_tst[,1], type = "l", col="dodgerblue", lwd=2,ylim=c(0.55,0.8), ylab = "Accuracy", xlab = "Epochs",
     main = "Average Test Accuracy for ADAM")
lines(ResultAcc_mean_tst[,2],lwd=2, col="purple")
lines(ResultAcc_mean_tst[,3], lwd=2,col="indianred")
lines(ResultAcc_mean_tst[,4], lwd=2,col="green")
lines(ResultAcc_mean_tst[,5], lwd=2,col="orange")
legend("topright", legend=c("5 neurons", "10 neurons","15 neurons", "20 neurons", "25 neurons"),
       col=c("dodgerblue", "purple","indianred","green","orange"), lwd = 2, cex=0.8, box.lwd=2)
grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted",
     lwd = par("lwd"), equilogs = TRUE)



# Train accuracy
ResultAcc_mean_tr<- matrix(ncol = 5, nrow = 100)
l=10 #number of experiments
h=1 
for (i in 1:100){
  
  ResultAcc_mean_tr[i,h] = mean(acc.tr[i,1:10])
  ResultAcc_mean_tr[i,h+1] = mean(acc.tr[i,11:20])
  ResultAcc_mean_tr[i,h+2] = mean(acc.tr[i,21:30])
  ResultAcc_mean_tr[i,h+3] = mean(acc.tr[i,31:40])
  ResultAcc_mean_tr[i,h+4] = mean(acc.tr[i,41:50])
  
}

plot(1:100,ResultAcc_mean_tr[,1], type = "l", col="dodgerblue", lwd=2,ylim=c(0.4,2), ylab = "Accuracy", xlab = "Epochs",
     main = "Average Train Accuracy for ADAM")
lines(ResultAcc_mean_tr[,2],lwd=2, col="purple")
lines(ResultAcc_mean_tr[,3], lwd=2,col="indianred")
lines(ResultAcc_mean_tr[,4], lwd=2,col="green")
lines(ResultAcc_mean_tr[,5], lwd=2,col="orange")
legend("bottomright", legend=c("5 neurons", "10 neurons","15 neurons", "20 neurons", "25 neurons"),
       col=c("dodgerblue", "purple","indianred","green","orange"), lwd = 2, cex=0.8, box.lwd=2)
grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted",
     lwd = par("lwd"), equilogs = TRUE)






                                   #### Learning Rate ####

train.loss <- matrix(ncol = 2, nrow = 50)
train.acc <- matrix(ncol = 2, nrow =50)
test.loss <- matrix(ncol = 2, nrow = 50)
test.acc <- matrix(ncol = 2, nrow = 50)
k = 1
loss.tr <- matrix(ncol = 50, nrow = 100)
loss.tst <- matrix(ncol = 50, nrow = 100)
acc.tr <- matrix(ncol = 50, nrow = 100)
acc.tst <- matrix(ncol = 50, nrow = 100)


for (learning.rate in c(0.1,0.2,0.3,0.5,1)) {
  
  for (i in 1:10) {
    
    model.lr <- keras_model_sequential() 
    model.lr %>% 
      layer_dense(units = 25, activation = 'relu', input_shape = ncol(X_train)) %>% 
      layer_dense(units = 2, activation = 'softmax')
    
    model.lr %>% compile(
      loss = 'binary_crossentropy',
      optimizer = optimizer_adam(
        lr = learning.rate,
        beta_1 = 0.9,
        beta_2 = 0.999,
        decay = 1e-4,
        amsgrad = FALSE,
        clipnorm = 1,
        clipvalue = 0.5),
      metrics = c('accuracy'))
    
    
    history.lr<- model.lr %>% fit(
      X_train, y_train, 
      epochs = 100, batch_size = 50, 
      validation_split = 0.3, shuffle=TRUE)
    
    loss.tr[,k] <- history.lr$metrics$loss
    loss.tst[,k] <- history.lr$metrics$val_loss
    acc.tr[,k] <- history.lr$metrics$accuracy
    acc.tst[,k] <- history.lr$metrics$val_accuracy
    
    results.train <- model.lr %>% evaluate(X_train, y_train)
    train.loss[k,1] <- learning.rate
    train.loss[k,2] <- results.train$loss
    
    train.acc[k,1] <-  learning.rate
    train.acc[k,2] <- results.train$accuracy
    
    results.test <- model.lr %>% evaluate(X_test, y_test)
    test.loss[k,1] <- learning.rate
    test.loss[k,2] <- results.test$loss
    
    test.acc[k,1] <- learning.rate
    test.acc[k,2] <- results.test$accuracy
    
    k = k+1
    
  }
  
}

# Mean and 95% CI
LR.trainLOSS<- c(CI(train.loss[1:10,2]),CI(train.loss[11:20,2]),
                 CI(train.loss[21:30,2]),CI(train.loss[31:40,2]),CI(train.loss[41:50,2]))
LR.testLOSS<- c(CI(test.loss[1:10,2]),CI(test.loss[11:20,2]),
                CI(test.loss[21:30,2]),CI(test.loss[31:40,2]),CI(test.loss[41:50,2]))
LR.trainACC <- c(CI(train.acc[1:10,2]),CI(train.acc[11:20,2]),
                 CI(train.acc[21:30,2]),CI(train.acc[31:40,2]),CI(train.acc[41:50,2]))
LR.testACC <- c(CI(test.acc[1:10,2]),CI(test.acc[11:20,2]),
                CI(test.acc[21:30,2]),CI(test.acc[31:40,2]),CI(test.acc[41:50,2]))

# Plots

# Training loss
ResultLoss_mean_tr<- matrix(ncol = 5, nrow = 100)
l=10 #number of experiments
h=1 
for (i in 1:100){
  
  ResultLoss_mean_tr[i,h] = mean(loss.tr[i,1:10])
  ResultLoss_mean_tr[i,h+1] = mean(loss.tr[i,11:20])
  ResultLoss_mean_tr[i,h+2] = mean(loss.tr[i,21:30])
  ResultLoss_mean_tr[i,h+3] = mean(loss.tr[i,31:40])
  ResultLoss_mean_tr[i,h+4] = mean(loss.tr[i,41:50])
  
}
plot(1:100,ResultLoss_mean_tr[,1], type = "l", col="dodgerblue", lwd=2,ylim=c(0,9), ylab = "Loss", xlab = "Epochs",
     main = "Average Training Loss for ADAM")
lines(ResultLoss_mean_tr[,2],lwd=2, col="purple")
lines(ResultLoss_mean_tr[,3], lwd=2,col="indianred")
lines(ResultLoss_mean_tr[,4], lwd=2,col="green")
lines(ResultLoss_mean_tr[,5], lwd=2,col="orange")
legend("topright", legend=c("lr: 0.1", "lr: 0.2","lr: 0.3", "lr: 0.5", "lr: 1"),
       col=c("dodgerblue", "purple","indianred","green","orange"), lwd = 2, cex=0.8, box.lwd=2)
grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted",
     lwd = par("lwd"), equilogs = TRUE)


## Test loss
ResultLoss_mean_tst<- matrix(ncol = 5, nrow = 100)
l=10 #number of experiments
h=1 
for (i in 1:100){
  
  ResultLoss_mean_tst[i,h] = mean(loss.tst[i,1:10])
  ResultLoss_mean_tst[i,h+1] = mean(loss.tst[i,11:20])
  ResultLoss_mean_tst[i,h+2] = mean(loss.tst[i,21:30])
  ResultLoss_mean_tst[i,h+3] = mean(loss.tst[i,31:40])
  ResultLoss_mean_tst[i,h+4] = mean(loss.tst[i,41:50])
  
}

plot(1:100,ResultLoss_mean_tst[,1], type = "l", col="dodgerblue", lwd=2,ylim=c(0,9), ylab = "Loss", xlab = "Epochs",
     main = "Average Test Loss for ADAM")
lines(ResultLoss_mean_tst[,2],lwd=2, col="purple")
lines(ResultLoss_mean_tst[,3], lwd=2,col="indianred")
lines(ResultLoss_mean_tst[,4], lwd=2,col="green")
lines(ResultLoss_mean_tst[,5], lwd=2,col="orange")
legend("topright", legend=c("lr: 0.1", "lr: 0.2","lr: 0.3", "lr: 0.5", "lr: 1"),
       col=c("dodgerblue", "purple","indianred","green","orange"), lwd = 2, cex=0.8, box.lwd=2)
grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted",
     lwd = par("lwd"), equilogs = TRUE)



# Test accuracy
ResultAcc_mean_tst<- matrix(ncol = 5, nrow = 100)
l=10 #number of experiments
h=1 
for (i in 1:100){
  
  ResultAcc_mean_tst[i,h] = mean(acc.tst[i,1:10])
  ResultAcc_mean_tst[i,h+1] = mean(acc.tst[i,11:20])
  ResultAcc_mean_tst[i,h+2] = mean(acc.tst[i,21:30])
  ResultAcc_mean_tst[i,h+3] = mean(acc.tst[i,31:40])
  ResultAcc_mean_tst[i,h+4] = mean(acc.tst[i,41:50])
  
}

plot(1:100,ResultAcc_mean_tst[,1], type = "l", col="dodgerblue", lwd=2,ylim=c(0.2,0.8), ylab = "Accuracy", xlab = "Epochs",
     main = "Average Test Accuracy for ADAM")
lines(ResultAcc_mean_tst[,2],lwd=2, col="purple")
lines(ResultAcc_mean_tst[,3], lwd=2,col="indianred")
lines(ResultAcc_mean_tst[,4], lwd=2,col="green")
lines(ResultAcc_mean_tst[,5], lwd=2,col="orange")
legend("bottomright", legend=c("lr: 0.1", "lr: 0.2","lr: 0.3", "lr: 0.5", "lr: 1"),
       col=c("dodgerblue", "purple","indianred","green","orange"), lwd = 2, cex=0.8, box.lwd=2)
grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted",
     lwd = par("lwd"), equilogs = TRUE)



# Train accuracy
ResultAcc_mean_tr<- matrix(ncol = 5, nrow = 100)
l=10 #number of experiments
h=1 
for (i in 1:100){
  
  ResultAcc_mean_tr[i,h] = mean(acc.tr[i,1:10])
  ResultAcc_mean_tr[i,h+1] = mean(acc.tr[i,11:20])
  ResultAcc_mean_tr[i,h+2] = mean(acc.tr[i,21:30])
  ResultAcc_mean_tr[i,h+3] = mean(acc.tr[i,31:40])
  ResultAcc_mean_tr[i,h+4] = mean(acc.tr[i,41:50])
  
}

plot(1:100,ResultAcc_mean_tr[,1], type = "l", col="dodgerblue", lwd=2,ylim=c(0.2,1), ylab = "Accuracy", xlab = "Epochs",
     main = "Average Train Accuracy for ADAM")
lines(ResultAcc_mean_tr[,2],lwd=2, col="purple")
lines(ResultAcc_mean_tr[,3], lwd=2,col="indianred")
lines(ResultAcc_mean_tr[,4], lwd=2,col="green")
lines(ResultAcc_mean_tr[,5], lwd=2,col="orange")
legend("bottomright", legend=c("lr: 0.1", "lr: 0.2","lr: 0.3", "lr: 0.5", "lr: 1"),
       col=c("dodgerblue", "purple","indianred","green","orange"), lwd = 2, cex=0.8, box.lwd=2)
grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted",
     lwd = par("lwd"), equilogs = TRUE)





                       #### Final comparison of the two models ####


### SGD: neurons= 25, lr = 0.1, mr = 0.9, 1 hidden layer
### ADAM: neurons= 25, lr = 0.1, 1 hidden layer

## SGD ##

sgd_model <- keras_model_sequential() 
sgd_model %>% 
  layer_dense(units = 25, activation = 'relu', input_shape = ncol(X_train),
              kernel_regularizer = regularizer_l2(l = 0.001)) %>%
  layer_dense(units = 2, activation = 'softmax')



sgd_model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_sgd(
    lr = 0.1,
    momentum = 0.9,
    decay = 1e-4,
    nesterov = FALSE,
    clipnorm = 1,
    clipvalue =0.5),
  metrics = c('accuracy'))


historySGD <- sgd_model %>% fit(
  X_train, y_train, 
  epochs = 100, batch_size = 50, 
  validation_split = 0.3, shuffle=TRUE)

results.trainSGD <- sgd_model %>% evaluate(X_train, y_train)
results.test <-sgd_model%>% evaluate(X_test, y_test)

probsSGD <-sgd_model %>% predict_proba(X_test) 
predSGD <- sgd_model %>% predict_classes(X_test)


## ADAM ##

adam_model <- keras_model_sequential() 
adam_model %>% 
  layer_dense(units = 25, activation = 'relu', input_shape = ncol(X_train),
              kernel_regularizer = regularizer_l2(l = 0.001)) %>% 
  layer_dense(units = 2, activation = 'softmax')


adam_model %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_adam(
    lr = 0.1,
    beta_1 = 0.9,
    beta_2 = 0.999,
    decay = 1e-4,
    amsgrad = FALSE,
    clipnorm = 1,
    clipvalue = 0.5),
  metrics = c('accuracy'))


historyADAM <- adam_model %>% fit(
  X_train, y_train, 
  epochs = 100, batch_size = 50, 
  validation_split = 0.3, shuffle=TRUE)

results.trainADAM <-adam_model %>% evaluate(X_train, y_train)
results.testADAM <- adam_model %>% evaluate(X_test, y_test)


probsADAM <- adam_model %>% predict_proba(X_test) 
predADAM <- adam_model %>% predict_classes(X_test)


# Comparison


comparison_models = data.frame(
  SGD_train= historySGD$metrics$loss,
  SGD_val = historySGD$metrics$val_loss,
  ADAM_train = historyADAM$metrics$loss,
  ADAM_val = historyADAM$metrics$val_loss)%>%rownames_to_column()

comparison_models$rowname = as.integer(comparison_models$rowname)
comparison_models = comparison_models%>%
  gather(key = "type", value = "value", -rowname)

# plot
ggplot(comparison_models, aes(x = rowname, y = value, color = type)) +
  geom_line(size=1) +
  xlab("epoch") +
  ylab("loss")


comparison_models = data.frame(
  SGD_train= historySGD$metrics$accuracy,
  SGD_val = historySGD$metrics$val_accuracy,
  ADAM_train = historyADAM$metrics$accuracy,
  ADAM_val = historyADAM$metrics$val_accuracy)%>%rownames_to_column()

comparison_models$rowname = as.integer(comparison_models$rowname)
comparison_models = comparison_models%>%
  gather(key = "type", value = "value", -rowname)

# plot
ggplot(comparison_models, aes(x = rowname, y = value, color = type)) +
  geom_line(size=1) +
  xlab("epoch") +
  ylab("accuracy")


library(pROC)
library(caret)
library(MLmetrics)

# F1 score SGD
F1_Score(test.set$parkinson, predSGD, positive = NULL) 

# F1 score ADAM
F1_Score(test.set$parkinson, predADAM, positive = NULL)

# Confusion matrix SGD
confusionMatrix(as.factor(predSGD),as.factor(test.set$parkinson))
  
# Confusion matrix ADAM
confusionMatrix(as.factor(predADAM),as.factor(test.set$parkinson))

# AUC for ADAM
auc(test.set$parkinson, predADAM) 

#AUC for SGD
auc(test.set$parkinson, predSGD)

