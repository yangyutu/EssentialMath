## margpoly.R
# From A First Course in Machine Learning, Chapter 3.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
# Marginal likelihood for model selection
rm(list=ls(all=TRUE))

## Generate the data
N = 100
x = sort(runif(N)*10-5)
noise_var = 150
t = 5*(x^3) - (x^2) + x
# Try adding and removing terms from this function, or changing term
# weights. e.g.
# t = 0.0005*(x^3) - (x^2) + x
t = t + rnorm(length(x))*sqrt(noise_var)

# Plot the data
plot(x,t,pch=16)

## Fit models of various orders
orders = 0:8
testx = seq(-5,5,0.01)
X = matrix(x^0,ncol=1)
testX = matrix(testx^0,ncol=1)
log_marg <- rep(NA, length(orders))
for(i in 1:length(orders)){
  ##
  si0 = diag(orders[i]+1)
  mu0 = rep(0,orders[i]+1)
  if(orders[i]>0){
    X = cbind(X,x^orders[i])
    testX = cbind(testX,testx^orders[i])
  }
  siw = solve((1/noise_var)*t(X)%*%X + solve(si0))
  muw = siw%*%((1/noise_var)*t(X)%*%t + solve(si0)%*%mu0)
  
  # Plot the data and mean function
  plot(x,t,pch=16,main=paste("Model order",orders[i]))
  lines(testx,testX%*%muw,col="red")
  
  # Compute the marginal likelihood
  margcov = noise_var*diag(N) + X%*%si0%*%t(X)
  margmu = X%*%mu0
  D = length(margmu)
  log_marg[i] = -(D/2)*log(2*pi) - 0.5*log(det(margcov))
  log_marg[i] = log_marg[i] - 0.5*t(t-margmu)%*%solve(margcov)%*%(t-margmu)
}

## Plot the marginal likelihoods
barplot(exp(log_marg),orders,xlab="Model Order",ylab="Marginal likelihood")








