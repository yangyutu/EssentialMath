## gpprior.R
# Plots realisations from a GP prior using an RBF covariance function
#
# From A First Course in Machine Learning
# Francois-Xavier Briol, 31/10/16 [f-x.briol@warwick.ac.uk]
rm(list=ls(all=TRUE))
require(MASS) ## may need install.packages("MASS")

## Define the x variable
# The first step is to define the x values of interest, we'll pick some
# random ones in the range -5 to 5
x = runif(200)*10-5
x = matrix(sort(x),nrow=200)

## Define the kernel parameters of interest
# We are using an RBF covariance function:
# 
# $$k(x_i,x_j) = \alpha\exp\left\{-\gamma(x_n-x_m)^2\right\}$$
# 
gamvals = c(0.05,0.1,0.5)
alpha = 1
## Loop over the gamma values
# at each stage, compute the covariance matrix and then sample 5 functions
# from the GP (i.e. 5 realisations from the N dimensional Gaussian with men
# zero and covariance equal to the covariance matrix.
N = length(x)
for(gam in gamvals){
  C = matrix(0,nrow=N,ncol=N)
  for(n in 1:N){
    for(n2 in 1:N){
      C[n,n2] = alpha*exp(-gam*(x[n]-x[n2])^2)
    }
  }
  # add some jitter for numerical stability
  C = C + (10^(-5))*diag(N)
  f = mvrnorm(n=5,mu=rep(0,N),Sigma=C)
  for(iter in 1:nrow(f)){
    if(iter==1){
      plot(x,f[iter,],type="l",lty=2,main=paste("Gamma =",gam),
           xlab="x",ylab="f(x)",ylim=c(-3,3))
    }
    if(iter!=1){
      points(x,f[iter,],type="l",lty=2)
    }
  }
}