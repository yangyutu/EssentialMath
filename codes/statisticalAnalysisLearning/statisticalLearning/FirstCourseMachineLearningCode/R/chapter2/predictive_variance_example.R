## predictive_variance_example.R
# From A First Course in Machine Learning, Chapter 2.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
# Predictive variance example
rm(list=ls(all=TRUE))
require(Hmisc) # might require to install package: install.packages("Hmisc")
require(MASS) # might require to install package: install.packages("MASS")

## Sample data from the true function
# $y = 5x^3-x^2+x$
N = 100 # Number of training points
x = sort(10*runif(N)-5) 
t = 5*(x^3) - (x^2) + x
noise_var = 300
t = t + rnorm(length(x))*sqrt(noise_var)

# Chop out some x data
pos = which(x>0 & x<2)
x = x[-pos]
t = t[-pos]

testx = seq(-5,5,0.1)

## Plot the data
plot(x,t)

## Fit models of various orders
orders = c(1:8)
for(i in 1:length(orders)){
  X = matrix(x^0)
  testX = matrix(testx^0)
  for(k in 1:orders[i]){
    X = cbind(X,x^k)
    testX = cbind(testX,testx^k)
  }
  w = solve(t(X)%*%X)%*%t(X)%*%t
  ss = (1/N)*(t(t)%*%t - t(t)%*%X%*%w)
  testmean = testX%*%w
  testvar = ss*diag(testX%*%solve(t(X)%*%X)%*%t(testX))
  # Plot the data and predictions
  plot(x,t,main=paste('Order',orders[i]),col="red",pch=16)
  errbar(testx,testmean, testmean+testvar, testmean-testvar,add=TRUE,cex=0.5)
}

## Plot sampled functions
orders = 1:8

for(i in 1:length(orders)){
  X = matrix(x^0,ncol=1)
  testX = matrix(testx^0,ncol=1)
  for(k in 1:orders[i]){
    X = cbind(X,x^k)
    testX = cbind(testX,testx^k)
  }
  w = solve(t(X)%*%X)%*%t(X)%*%t
  ss = as.numeric((1/N)*(t(t)%*%t - t(t)%*%X%*%w))
  ## Sample functions by sampling realisations of w from a Gaussian with
  # $\mu = \hat{\mathbf{w}},~~\Sigma =
  # \sigma^2(\mathbf{X}^T\mathbf{X})^{-1}$
  covw = ss*solve(t(X)%*%X)
  wsamp = mvrnorm(n = 10, mu=w, Sigma=covw)
  testmean = testX%*%t(wsamp)
  
  #Plot the data and functions
  plot(x,t,ylim=c(-600,600),main=paste("Order",orders[i]),col="red",pch=16)
  for(sample in 1:ncol(testmean)){
    points(testx,testmean[,sample],type="l")
  }
}

