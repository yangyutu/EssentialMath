## gpclass.R
# Performs binary GP classification with one dimensional data
#
# From A First Course in Machine Learning
# Francois-Xavier Briol, 31/10/16 [f-x.briol@warwick.ac.uk]
rm(list=ls(all=TRUE))
require(MASS) ## may need install.packages("MASS")

## Binary GP classification (1D) using the Laplace approximation

N = 10;
x = runif(N)
x = sort(x)

## Set the covariance parameters
gamma = 10
alpha = 10

## Compute the covariance matrix (training)
C = matrix(nrow=N,ncol=N)
for(n in 1:N){
  for(m in 1:N){
    C[n,m] = alpha*exp(-gamma*(x[n]-x[m])^2)
  }
}

## Generative model - Figure 8.18 in book
# Sample a function from the GP prior
f = mvrnorm(n = 1,mu=rep(0,N),Sigma=C)

# Plot the realisations of the function
plot(x,f,xlab="x",ylab="f(x)",pch=17,col="red")
yl = par("usr")[c(3,4)];
# Plot the data points along the x axis
points(x,rep(yl[1]+0.5,N),pch=16)

# Convert the function values to probabilities via a sigmoid function
p = 1/(1+exp(-f))
# Plot the probabilities
yl = c(-0.02,1)
plot(x,p,pch=17,col="red",ylim=yl)

# Get some random numbers
u = runif(N)
t = matrix(0,nrow=N,ncol=1)
# Set the target class to 1 if the random value is less than the probability
t[u<=t(p)] = 1
# Plot the data again, coloured according to class
pos = which(t==0)
points(x[pos],rep(yl[1]+0.01,length(pos)),ylim=yl,xlim=c(0,1),
     pch=16,xlab="x",ylab="P(T=1|x)")
pos = which(t==1)
points(x[pos],rep(yl[1]+0.01,length(pos)))


# Inference of f
# We will start with a point estimate obtained through numerical
# optimisation

# Newton-Rapshon procedure for finding the MAP estimate of f
# Initialise f as a vector of zeros
f = matrix(0,nrow=N,ncol=1)
# Pre-compute the inverse of C (for efficiency)
invC = solve(C)

# Newton-Raphson procedure (p.300)
max_its = 10
it = 1
allf = matrix(0,nrow=max_its,ncol=N)
allf[1,] = t(f)
while(it < max_its){
  g = 1/(1+exp(-f))
  gradient = -invC%*%f + t - g;
  hessian = -invC - diag(as.vector(g*(1-g)),nrow(g))
  f = f - solve(hessian)%*%gradient
  it = it + 1
  allf[it,] = t(f)
}
hessian = -invC - diag(as.vector(g*(1-g)),nrow(g))

# Plot the inferred optimised f values
plot(x,f,xlim=c(0,1),pch=17,col="red")
# plot the data
pos = which(t==0)
yl = par("usr")[c(3,4)]
points(x[pos],rep(yl[1]+0.03,length(pos)),pch=16)
pos = which(t==1)
points(x[pos],rep(yl[1]+0.03,length(pos)))

# Plot the evolution of the f values through the optimisation
plot(1:5,allf[1:5,1],xlab="Iteration",ylab="f",type="l",ylim=range(allf))
for(iter in 2:N){
  points(1:5,allf[1:5,iter],xlab="Iteration",ylab="f",type="l")
}

## Predictions with the point estimate
# Define some test points for vidualisation and compute the test covariance
testx = matrix(seq(0,1,0.01),ncol=1)
Ntest = nrow(testx)
# Compute the required covariance functions
R = matrix(nrow=N,ncol=Ntest)
for(n in 1:N){
  for(m in 1:Ntest){
    R[n,m] = alpha*exp(-gamma*(x[n]-testx[m])^2)
  }
}

Cstar = matrix(nrow=Ntest,ncol=Ntest)
for(n in 1:Ntest){
  for(m in 1:Ntest){
    Cstar[n,m] = alpha*exp(-gamma*(testx[n]-testx[m])^2)
  }
}

# Compute the latent function at the test points
fs = t(R)%*%invC%*%f

# Plot the predictive latent function
plot(as.vector(testx),as.vector(fs),type="l",xlab="x",ylab="f(x)")
pos = which(t==0)
yl = par("usr")[c(3,4)]
points(x[pos],f[pos],pch=16)
pos = which(t==1)
points(x[pos],f[pos])

# Plot the predictive probabilities
plot(testx,1./(1+exp(-fs)),type="l",
     xlab="x",ylab="P(T=1|x)",ylim=c(-0.03,1))
pos = which(t==0)
points(x[pos],1./(1+exp(-f[pos])),pch=16)
pos = which(t==1)
points(x[pos],1./(1+exp(-f[pos])))

## Propogating uncertainty through the sigmoid function
# In the previous example, we simply pushed the predictive mean through the
# sigmoid function. We can also account for the predictive variance. Here
# we do this by generating lots of samples from the latent function,
# pushing them all through the sigmoid function and then taking the mean

# Compute the predictive variance
predvar = diag(Cstar - t(R)%*%invC%*%R)

# Watch out for really small values
predvar[which(predvar<1e-6)] = 1e-6

# Generare lots of samples and then average
Nsamples = 10000
u = matrix(rnorm(Ntest*Nsamples),nrow=Ntest,ncol=Nsamples)*
  matrix(rep(sqrt(predvar),Nsamples),ncol=Nsamples,byrow=F) + matrix(rep(fs,Nsamples),ncol=Nsamples,byrow=F)
pavg = rowMeans(1/(1+exp(-u)))

# Plot the resulting predictive probabilities
plot(as.vector(testx),pavg,type="l",xlab="x",ylab="P(T=1|x)",ylim=c(-0.02,1))
pos = which(t==0)
points(x[pos],1./(1+exp(-f[pos])),pch=16)
pos = which(t==1)
points(x[pos],1./(1+exp(-f[pos])))

## Plot the marginal posterior at the training points
# Plot the posterior of the latent variables at the training points (mean
# plus and minus standard deviation)

# Compute the posterior covariance
covf = -solve(hessian)
post_sd = sqrt(diag(covf))
pos = which(t==0)
plot(x[pos],f[pos],xlab="x",ylab="f(x)",pch=16,xlim=c(0,1),ylim=c(-2,4))
arrows(x[pos], f[pos]-post_sd[pos], x[pos], f[pos]+post_sd[pos], length=0.05, angle=90, code=3,pch=16)
pos = which(t==1);
points(x[pos],f[pos])
arrows(x[pos], f[pos]-post_sd[pos], x[pos], f[pos]+post_sd[pos], length=0.05, angle=90, code=3)

## And the test predictions
# Add the predictive mean and variance, and some samples to the plots.

# Compute the predictive mean and covariance
pred_mu = t(R)%*%invC%*%f
pred_cov = Cstar - t(R)%*%solve(C,R) + t(R)%*%solve(C,covf)%*%invC%*%R
pred_cov = pred_cov + (10^(-4))*diag(Ntest)

# Sample some predictive functions and plot them
samps = gausssamp(pred_mu,pred_cov,N=10);
plot(as.vector(testx),samps[1,],type="l",lty=2,ylim=c(-5,7),xlab="x",ylab="f(x)")
for(iter in 2:10){
  points(as.vector(testx),samps[iter,],type="l",lty=2)
}

# Plot the predictive mean and variance
points(as.vector(testx),pred_mu,type="l",lwd=2)
sd = sqrt(diag(pred_cov))
points(testx,pred_mu + sd,type="l",lwd=2)
points(testx,pred_mu - sd,type="l",lwd=2)

# Plot the values at the data
pos = which(t==0);
points(x[pos],f[pos],pch=16,col="red")
arrows(x[pos], f[pos]-post_sd[pos], x[pos], f[pos]+post_sd[pos], length=0.05, angle=90, code=3,lwd=2)
pos = which(t==1);
points(x[pos],f[pos],pch=16,col="blue")
arrows(x[pos], f[pos]-post_sd[pos], x[pos], f[pos]+post_sd[pos], length=0.05, angle=90, code=3,lwd=2)

# Turn each sample into a probability to plot
plot(testx,1/(1+exp(-t(samps[1,]))),type="l",xlab="x",ylab="P(T=1|x)",ylim=c(0,1))
for(iter in 2:10){
  points(testx,1/(1+exp(-t(samps[iter,]))),type="l")
}
