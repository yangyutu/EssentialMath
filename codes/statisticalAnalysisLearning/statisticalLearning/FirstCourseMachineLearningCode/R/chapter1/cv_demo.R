## cv_demo.R
# From A First Course in Machine Learning, Chapter 1.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
# Demonstration of cross-validation for model selection
rm(list=ls(all=TRUE))

## Generate some data
# Generate x between -5 and 5
N = 100
x = 10*runif(N) - 5
t = 5*(x^3)  - (x^2) + x + 150*rnorm(length(x))
testx = seq(-5,5,0.01) # Large, independent test set
testt = 5*(testx^3) - (testx^2) + testx + 150*rnorm(length(testx))

## Run a cross-validation over model orders
maxorder = 7
X = matrix(x^0,ncol=1)
testX = matrix(testx^0,ncol=1)
K = 10 # K-fold CV
sizes = rep(floor(N/K),K)
sizes[length(sizes)] = sizes[length(sizes)] + N - sum(sizes)
csizes = c(0,cumsum(sizes))
cv_loss <- matrix(nrow=K,ncol=maxorder+1)
ind_loss <- matrix(nrow=K,ncol=maxorder+1)
train_loss <- matrix(nrow=K,ncol=maxorder+1)

# Note that it is often sensible to permute the data objects before
# performing CV.  It is not necessary here as x was created randomly.  If
# it were necessary, the following code would work:
  # order = sample(N)
# x = x[order] Or: X = X[order,] if it is multi-dimensional.
# t = t[order]

for(k in 0:maxorder){
  if(k>=1){
    X = cbind(X,x^k)
    testX = cbind(testX,testx^k)
  }
  for(fold in 1:K){
    # Partition the data
    # foldX contains the data for just one fold
    # trainX contains all other data
    foldX = as.matrix(X[(csizes[fold]+1):csizes[fold+1],])
    trainX = X
    trainX = as.matrix(trainX[-((csizes[fold]+1):csizes[fold+1]),])
    foldt = as.matrix(t[(csizes[fold]+1):csizes[fold+1]])
    traint = t
    traint = as.matrix(traint[-((csizes[fold]+1):csizes[fold+1])])
    
    w = solve(t(trainX)%*%trainX)%*%t(trainX)%*%traint
    fold_pred = foldX%*%w
    cv_loss[fold,k+1] = mean((fold_pred-foldt)^2)
    ind_pred = testX%*%w
    ind_loss[fold,k+1] = mean((ind_pred - testt)^2)
    train_pred = trainX%*%w
    train_loss[fold,k+1] = mean((train_pred - traint)^2)
  }
}

## Plot the results
par(mfrow=c(1,3))
plot(0:maxorder,colMeans(cv_loss),type="l",xlab="Model Order",ylab="Loss",main="CV Loss",col="blue")
plot(0:maxorder,colMeans(train_loss),type="l",xlab="Model Order",ylab="Loss",main="Train Loss",col="blue")
plot(0:maxorder,colMeans(train_loss),type="l",xlab="Model Order",ylab="Loss",main="Independent Test Loss",col="blue")
par(mfrow=c(1,1))

