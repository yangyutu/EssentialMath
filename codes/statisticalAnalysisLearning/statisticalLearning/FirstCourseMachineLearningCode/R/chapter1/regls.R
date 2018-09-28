## regls.R
# From A First Course in Machine Learning, Chapter 1.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
rm(list=ls(all=TRUE))

# An example of regularised least squares
# Data is generated from a linear model and then a fifth order polynomial is fitted.
# The objective (loss) function that is minimisied is
#  
# $${\cal L} = \lambda \mathbf{w}^T\mathbf{w} + \sum_{n=1}^N (t_n - f(x_n;\mathbf{w}))^2  $$

## Generate the data
x = seq(0,1,0.2)
y = (2*x)-3

## Create targets by adding noise
noisevar = 3
t = y + sqrt(noisevar)*rnorm(length(x))

## Plot the data
par(mar = c(5, 4, 4, 10))
plot(x,t,xlab="x",ylab="f(x)",xlim=c(-0.1,1.1),ylim=c(-5,1))

## Build up the data so that it has up to fifth order terms
testx = seq(0,1,0.01)
X = matrix(x^0)
testX = matrix(testx^0)
for(k in 1:5){
    X = cbind(X,x^k)
    testX = cbind(testX,testx^k)
}

## Fit the model with different values of the regularization parameter
## 
# $$\lambda$$
lam = c(0,1e-6,1e-2,1e-1)
for(l in 1:length(lam)){
  lambda = lam[l]
  N = length(x)
  w = solve(t(X)%*%X + N*lambda*diag(dim(X)[2]))%*%t(X)%*%t
  points(testx,testX%*%w,type="l",col=l)
}
par(xpd=T)
legend(x=1.2,y=-2,legend=c('lambda=0','lambda=10^-6','lambda=10^-2','lambda=10^-1'),
       lty=c(1,1,1,1),col=c("black","red","green","blue"))
par(xpd=F,mar = c(5, 4, 4, 2))

