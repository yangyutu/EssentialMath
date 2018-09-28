## gpclasslaplace.R
# Demomnstrates making predictions using the Laplace approximation for GP
# classification for 1- and 2-D data.
#
# From A First Course in Machine Learning
# Francois-Xavier Briol, 31/10/16 [f-x.briol@warwick.ac.uk]

## Full Laplace predictions - 1 dimensional data
# We can now look at the full Laplace approximation. Here we sample from
# the full multivariate Gaussian defined by the Laplace approximation
# Note that the first cell duplicates lots of things from gpclass.R

rm(list=ls(all=TRUE))
require(MASS) ## may need install.packages("MASS")

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

# Convert the function values to probabilities via a sigmoid function
p = 1/(1+exp(-f))
# Plot the probabilities
yl = c(-0.02,1)
u = runif(N)
t = matrix(0,nrow=N,ncol=1)
t[u<=t(p)] = 1

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

# Compute the posterior covariance
covf = -solve(hessian)

# Compute the predictive mean and covariance
pred_mu = t(R)%*%invC%*%f
pred_cov = Cstar - t(R)%*%solve(C,R) + t(R)%*%solve(C,covf)%*%invC%*%R
pred_cov = pred_cov + (10^(-4))*diag(Ntest)

## Generate samples from the Laplace approximation and make predictions
# Generate 1000 samples from the Laplace approximation
samps = gausssamp(pred_mu,pred_cov,1000) ## may need to load gausssamp.R
# Convert into probabilities and compute mean
p = 1/(1+exp(-samps))
p = colMeans(p)
# Plot the Laplace predictions
plot(testx,p,type="l",xlab="x",ylab="P(T=1|x)",ylim=c(0,1.05))

# Plot the point approximation for comparison
points(testx,1/(1+exp(-fs)),type="l",lty=2)
pos = which(t==0)
points(x[pos],1./(1+exp(-f[pos])),pch=16)
pos = which(t==1);
points(x[pos],1./(1+exp(-f[pos])))

## Laplace approximation for the 2D example
# Again, the first cell is setting things up, taken from class2d.R


## create some random data and then change the means of the two classes to
# separate them
x = matrix(rnorm(40),ncol=2)
x[1:10,] = x[1:10,] - 2
x[11:20,] = x[11:20,] +2
t = rep(c(0,1),each=10)


## Set the GP hyperparameters and compute the covariance function
alpha = 10
gamma = 0.1
N = nrow(x)
C = matrix(0,ncol=N,nrow=N)
for(n in 1:N){
  for(m in 1:N){
    C[n,m] = alpha*exp(-gamma*sum((x[n,]-x[m,])^2))
  }
}


# Newton-raphson procedure to optimise the latent function values
# Initialise all to zero
f = matrix(0,nrow=N,ncol=1)
allf = matrix(0,nrow=6,ncol=N)

# Pre-compute the inverse of C
invC = solve(C)
for(iteration in 2:6){
  g = 1/(1+exp(-f))
  gra = t - g - (invC%*%f)
  H = -diag(as.vector(g*(1-g)),nrow = nrow(g)) - invC
  f = f - solve(H)%*%gra
  allf[iteration,] = t(f)
}
H = -diag(as.vector(g*(1-g)),nrow = nrow(g)) - invC

# Plot the evolution of the f values

## Visualise the predictive function via a large grid of test points
# Create the grid
gridseq = seq(from=-5.5,to=5.5,by=0.25)
mesh=meshgrid(gridseq)
X = mesh$X; Y = mesh$Y
testN = prod(dim(X))
testX = cbind(as.vector(X),as.vector(Y))
# Create the test covariance function
R = matrix(0,nrow=N,ncol=testN)
for(n in 1:N){
  for(m in 1:testN){
    R[n,m] = alpha*exp(-gamma*sum((x[n,] - testX[m,])^2))
  }
}
    
## Full Laplace approximation - compute the test covariance
Cstar = matrix(0,nrow=testN,ncol=testN)
for(n in 1:testN){
  for(m in 1:testN){
    Cstar[n,m] = alpha*exp(-gamma*sum((testX[n,] - testX[m,])^2))
  }
}
covf = -solve(H)
pred_mu = t(R)%*%invC%*%f
pred_cov = Cstar - t(R)%*%solve(C,R) + t(R)%*%solve(C,covf)%*%invC%*%R
pred_cov = pred_cov + (10^(-4))*diag(testN)

## Generate 3 sample functions from the Laplace and plot the resulting decision boundaries
n_samps = 3
samps = gausssamp(pred_mu,pred_cov,n_samps) ## may require to load gausssamp.R
p = 1/(1+exp(-samps))
dev.off()
filled.contour(gridseq,gridseq,matrix(p[1,],ncol=length(gridseq),byrow=T),
               xlab="x_1",ylab="x_2",main="Sample 1",
               plot.axes={points(x[,1],x[,2],pch=16)})
filled.contour(gridseq,gridseq,matrix(p[2,],ncol=length(gridseq),byrow=T),
               xlab="x_1",ylab="x_2",main="Sample 2",
               plot.axes={points(x[,1],x[,2],pch=16)})
filled.contour(gridseq,gridseq,matrix(p[3,],ncol=length(gridseq),byrow=T),
               xlab="x_1",ylab="x_2",main="Sample 3",
               plot.axes={points(x[,1],x[,2],pch=16)})

## Average over lots of samples from the Laplace
n_samps = 1000;
samps = gausssamp(pred_mu,pred_cov,n_samps)
p = 1./(1+exp(-samps))
avgp = colMeans(p,1);
filled.contour(gridseq,gridseq,matrix(avgp,ncol=length(gridseq),byrow=T),
               xlab="x_1",ylab="x_2",main="Averaged Laplace",
               plot.axes={points(x[,1],x[,2],pch=16)})

