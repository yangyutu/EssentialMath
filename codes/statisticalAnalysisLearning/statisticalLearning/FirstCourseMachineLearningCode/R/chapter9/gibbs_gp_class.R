## gibbs_gp_class.R
# Using gibbs sampling for binary GP classification Gaussian
#
# From A First Course in Machine Learning
# Francois-Xavier Briol, 31/10/16 [f-x.briol@warwick.ac.uk]
#
rm(list=ls(all=TRUE))
require(MASS) ## may need install.packages("MASS")

## Generate a dataset
x = matrix(sort(runif(10)),ncol=1)
N = length(x)
t = matrix(rep(c(0,1),each=5),ncol=1)
## Plot the data
pos = which(t==0)
plot(x[pos],t[pos],pch=16,xlim=c(0,1),ylim=c(-0.05,1.05))
pos = which(t==1)
points(x[pos],t[pos],pch=16,col="red")

## Define a test set for visualisation
testx = matrix(seq(0,1,0.01),ncol=1)
Ntest = length(testx)
# Set the number of Gibbs samples to draw
S = 10000
# Initialise the auxiliary variables, y
y = rnorm(N)
y[t==0] = y[t==0] - 2
y[t==1] = y[t==1] + 2
y = matrix(y,ncol=1)

# Objects to store the sampled values
allF = matrix(0,nrow=S,ncol=N)
allT = matrix(0,nrow=S,ncol=Ntest)
allFstar = matrix(0,nrow=S,ncol=Ntest)
trainTall = matrix(0,nrow=S,ncol=N)

## Set the hyperparameters and create the covariance matrices
gamma = 10.0
alpha = 1
C = matrix(0,nrow=N,ncol=N)
R = matrix(0,nrow=N,ncol=Ntest)
for(n in 1:N){
  for(m in 1:N){
    C[n,m] = alpha*exp(-gamma*(x[n]-x[m])^2)
  }
  for(m in 1:Ntest){
    R[n,m] = alpha*exp(-gamma*(x[n] - testx[m])^2)
  }
}

## Main sampling loop
sif = solve(solve(C) + diag(N))
Ci = solve(C)
# Loop over the number of desired samples
for(s in 1:S){
  
  # Update f
  f = mvrnorm(n = 1, mu=sif%*%y,Sigma=sif)
  allF[s,] = f
  
  # update y (using rejection sampling)
  for(n in 1:N){
    finished = 0
    while(!finished){
      y[n] = rnorm(1) + f[n]
      if(y[n]*(2*t[n]-1) > 0){
        finished = 1
      }
    }
  }

  # Sample a predictive function f^*
  f_star_mu = t(R)%*%Ci%*%f
  f_star_ss = alpha - diag(t(R)%*%solve(C,R))
  
  # Look out for -ve values caused by 
  f_star = matrix(rnorm(Ntest),ncol=1)*sqrt(f_star_ss) + f_star_mu
  allFstar[s,] = t(f_star)
  
  # Use this to make (and store) some predictions
  y_star = matrix(rnorm(Ntest),ncol=1) + f_star
  allT[s,] = t(y_star>0)
  tempy = matrix(rnorm(N),ncol=1) + matrix(f,ncol=1)
  trainTall[s,] = t(tempy>0)
}

## Plot the posterior f values
pos = which(t==0)
plot(x[pos],colMeans(allF[,pos]),xlim=c(0,1),ylim=c(-3,3),pch=16)
arrows(x[pos], colMeans(allF[,pos])-apply(allF[,pos],2,sd), x[pos], colMeans(allF[,pos])+apply(allF[,pos],2,sd), length=0.05, angle=90, code=3,pch=16)
pos = which(t==1)
points(x[pos],colMeans(allF[,pos]),xlim=c(0,1),ylim=c(-3,3),pch=16,col="red")
arrows(x[pos], colMeans(allF[,pos])-apply(allF[,pos],2,sd), x[pos], colMeans(allF[,pos])+apply(allF[,pos],2,sd), length=0.05, angle=90, code=3,pch=16)

## Plot 10 randomly selected samples of f from the posterior
nplot = 10
order = sample(1:S,S)
plot(as.vector(testx),allFstar[order[1],],type="l",lty=2,
     xlim=c(0,1),ylim=c(-4,4))
for(iter in 2:nplot){
  points(as.vector(testx),allFstar[order[iter],],type="l",lty=2)
}
# Add the mean and std to the plot
points(testx,colMeans(allFstar),type="l",lwd=2)
points(testx,colMeans(allFstar)+apply(allFstar,2,sd),type="l",lwd=2)
points(testx,colMeans(allFstar)-apply(allFstar,2,sd),type="l",lwd=2)
pos = which(t==0)
points(x[pos],colMeans(allF[,pos]),xlim=c(0,1),ylim=c(-3,3),pch=16)
arrows(x[pos], colMeans(allF[,pos])-apply(allF[,pos],2,sd), x[pos], colMeans(allF[,pos])+apply(allF[,pos],2,sd), length=0.05, angle=90, code=3,pch=16)
pos = which(t==1)
points(x[pos],colMeans(allF[,pos]),xlim=c(0,1),ylim=c(-3,3),pch=16,col="red")
arrows(x[pos], colMeans(allF[,pos])-apply(allF[,pos],2,sd), x[pos], colMeans(allF[,pos])+apply(allF[,pos],2,sd), length=0.05, angle=90, code=3,pch=16)

## Plot the mean predictive probabilities
plot(testx,colMeans(allT),type="l",xlim=c(0,1))
traint = colMeans(trainTall)
pos = which(t==0)
points(x[pos],traint[pos],pch=16)
pos = which(t==1)
points(x[pos],traint[pos],pch=16,col="red")
