## lapexample.R
# From A First Course in Machine Learning, Chapter 4.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
# The Laplace approximation to a gamma density
rm(list=ls(all=TRUE))

## Define the gamma parameters
alpha = 20
beta = 0.5

# Find the mode
y_hat = (alpha-1)/beta
# Find the variance
ss = (alpha-1)/(beta^2)

## Plot the gamma and the approximate Gaussian
y = seq(0,100,0.01)
plot(y,dgamma(y,shape=alpha,rate=beta),type="l",xlab="y",ylab="p(y)") 
points(y,dnorm(y,y_hat,sqrt(ss)),type="l", lty=2)
legend(x=55,y=0.045,legend=c('Gamma','Laplace'),lty=c(1,2))

## Second Example
alpha = 2
beta = 100

# Find the mode
y_hat = (alpha-1)/beta
# Find the variance
ss = (alpha-1)/(beta^2)

## Plot the gamma and the approximate Gaussian
y = seq(0,0.1,0.0001)
plot(y,dgamma(y,shape=alpha,rate=beta),type="l",xlab="y",ylab="p(y)",ylim=c(0,40)) 
points(y,dnorm(y,y_hat,sqrt(ss)),type="l", lty=2)
legend(x=0.03,y=40,legend=c('Gamma','Laplace'),lty=c(1,2))

