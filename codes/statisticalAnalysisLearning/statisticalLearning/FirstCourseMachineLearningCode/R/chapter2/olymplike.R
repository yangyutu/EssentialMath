## olymplike.R
# From A First Course in Machine Learning, Chapter 2.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
# Likelihood increases as model complexity increases - example
rm(list=ls(all=TRUE))

## Load the Olympic data
setwd("~/data/olympics") ## Might need to change the path here
male100 = read.csv(file="male100.csv",header=FALSE)

x = male100[,1]
t = male100[,2]

# Rescale x for numerical stability
x = x - x[1]
x = x/30

## Fit different order models with maximum likelihood
# $\hat{\mathbf{w}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{t}$
##
# $\hat{\sigma^2} = \frac{1}{N}(\mathbf{t}^T\mathbf{t} - \mathbf{t}^T\mathbf{X}\hat{\mathbf{w}})$
orders = 1:8
X = matrix(x^0,ncol=1)
N = length(x)
log_like <- rep(NA,length(orders))
for(i in 0:length(orders)){
  if(i>0){X = cbind(X,x^orders[i])}
  w = solve(t(X)%*%X)%*%t(X)%*%t
  ss = (1/N)*(t(t)%*%t - t(t)%*%X%*%w)
  log_like[i] = sum(log(dnorm(t,X%*%w,sqrt(ss))))
}
        
## Plot the model order versus the (log) likelihood
plot(orders,log_like,xlab="Model order",ylab="Log likelihood",pch=16)
points(orders,log_like,type="l")
