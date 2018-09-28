## gpprednoise.R
# Performs noisy GP predictions
#
# From A First Course in Machine Learning
# Francois-Xavier Briol, 31/10/16 [f-x.briol@warwick.ac.uk]
#
# Note that this is identical to gppred.m with the addition of a diagonal
# noise component to the training covariance matrix

rm(list=ls(all=TRUE))
require(MASS) ## may need install.packages("MASS")

## Set the kernel parameters
# Try varying these to see the effect
gamma = 0.5
alpha = 1.0
sigma_sq=0.1

## Define x
# In this example, we use uniformly spaced x values
x = matrix(seq(-5,5,2),nrow=1)
N = ncol(x)

## Create the training covariance matrix
# Firstly for the training data
C = matrix(0,ncol=N,nrow=N)
C = alpha*exp(-gamma*(matrix(rep(x,N),ncol=N) - matrix(rep(x,N),ncol=N,byrow=T))^2)+sigma_sq*diag(N)

# Generate a true function from the GP prior with zero mean
# Define the GP mean
mu = matrix(rep(0,N),nrow=1)
# Sample a function
true_f = mvrnorm(n=1,mu=mu,Sigma=C) 
## Define the test points - again on a uniform grid
testx = matrix(c(-4,-2,0,2,4),nrow=1)

## Plot the training data and show the position of the test points
xl = c(-6,6)
yl = range(c(true_f+0.25,true_f-0.25))
plot(x,true_f,xlab="x",ylab="f(x)",pch=16,xlim=xl,ylim=yl,col="red")
# Draw dashed lines at the test points 
for(i in testx){
  abline(v=i,lty=2)
}

## Compute the test covariance
# We need two matrices, $\mathbf{C}^*$ and $\mathbf{R}$ (see page 285)
Ntest = length(testx)
# The train by test matrix
R = alpha*exp(-gamma*(matrix(rep(t(x),Ntest),ncol=Ntest) - matrix(rep(testx,N),nrow=N,byrow=N))^2)
# The test by test matrix
Cstar = alpha*exp(-gamma*(matrix(rep(testx,Ntest),nrow=Ntest,byrow=T)-matrix(rep(t(testx),Ntest),ncol=Ntest))^2)

## Compute the mean and covariance of the predictions
# This uses equation 8.3, p.285
# Plot the training data
plot(x,true_f,xlim=xl,ylim=yl,xlab="x",ylab="f(x)",pch=16,col="red")
# Compute the mean and covariance
pred_mu = t(R)%*%solve(C)%*%matrix(true_f,ncol=1)
pred_cov = Cstar - t(R)%*%solve(C)%*%R
# Extract the standard deviations at the test points (square root of the 
# diagonal elements of the covariance matrix)
pred_sd = sqrt(diag(pred_cov))
# Plot the predictions as error bars
points(testx,pred_mu,pch=16)
arrows(testx,pred_mu-pred_sd,testx,pred_mu+pred_sd, length=0.05, angle=90, code=3,pch=16)

## Plot a smooth predictive function with a one sd window,
# plot the data
plot(x,true_f,xlim=xl,ylim=yl,xlab="x",ylab="f(x)",pch=16,col="red",lwd=2)
# Define a new set of test x values
testx = seq(-5,5,0.1)
Ntest = length(testx)

# Compute R and Cstar again for the new test points 
R = exp(-gamma*(matrix(rep(t(x),Ntest),ncol=Ntest) - matrix(rep(testx,N),nrow=N,byrow=T))^2)
Cstar = exp(-gamma*(matrix(rep(testx,Ntest),nrow=Ntest,byrow=T) - matrix(rep(t(testx),Ntest),ncol=Ntest))^2)

# Compute the predictive mean and covariance
pred_mu =t(R)%*%solve(C)%*%matrix(true_f,ncol=1)
pred_cov = Cstar - t(R)%*%solve(C)%*%R

# Plot the predicted mean and plus and minus one sd
points(testx,pred_mu,type="l",lwd=2)
points(testx,pred_mu + sqrt(abs(diag(pred_cov))),type="l",lty=2,lwd=2)
points(testx,pred_mu - sqrt(abs(diag(pred_cov))),type="l",lty=2,lwd=2)

## Finally, plot some sample functions drawn from the predictive distribution
# Plot the data
plot(x,true_f,xlim=xl,ylim=yl,xlab="x",ylab="f(x)",pch=16,col="red",lwd=2)
# Draw 10 samples from the multivariate gaussian
f_samps = mvrnorm(n=10,mu=pred_mu,Sigma=pred_cov)
# plot them
for(iter in 1:10){
  points(testx,f_samps[iter,],type="l",lty=2)
}

