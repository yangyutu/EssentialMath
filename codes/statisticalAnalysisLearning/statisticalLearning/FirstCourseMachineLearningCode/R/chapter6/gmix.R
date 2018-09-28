## gmix.R
# From A First Course in Machine Learning, Chapter 6.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
# Fitting a Gaussian mixture
rm(list=ls(all=TRUE))

## Load the data
setwd("~/data/kmeansdata") ## Might need to change the path here
X = as.matrix(read.csv(file="X.csv",header=FALSE))

## Plot the data
plot(X[,1],X[,2],pch=16,xlim=c(-6,6),ylim=c(-6,6))

## Initilaise the mixture
K = 3 # Try changing this
means = matrix(rnorm(K*2),nrow=K,ncol=2)
covs = list()
for(k in 1:K){
  covs[[k]] = runif(1)*diag(2)
}
priors = rep(1/K,K)

## Run the algorithm
MaxIts = 100
N = nrow(X)
q = matrix(0,nrow=N,ncol=K)
D = ncol(X)
cols = c("red","darkgreen","blue")
plotpoints = c(1:10,seq(12,30,2),40,50)
B = -Inf;
converged = FALSE
it = 0
tol = 1e-2
temp = matrix(0,nrow=N,ncol=K)

while(converged == FALSE && it<=MaxIts){
  it = it + 1
  # Update q
  for(k in 1:K){
    const = -(D/2)*log(2*pi) - 0.5*log(det(covs[[k]]))
    Xm = X - matrix(rep(means[k,],N),nrow=N,byrow=TRUE)
    temp[,k] = const - 0.5*diag(Xm%*%solve(covs[[k]])%*%t(Xm))
  }
  
  # Compute the Bound on the likelihood
  if(it>1){
    B = c(B,sum(q*log(matrix(rep(priors,N),nrow=N,byrow=TRUE))))
    B[it] = B[it] + sum(q*temp) - sum(q*log(q))
    if(abs(B[it]-B[it-1])<tol){
      converged = TRUE
    }
  }
  
  temp = temp + matrix(rep(priors,N),nrow=N,byrow=TRUE)
  q = exp(temp - matrix(rep(apply(X=temp,MARGIN=1,FUN=max),K),ncol=K,byrow=FALSE))
  
  # Minor hack for numerical issues - stops the code crashing when
  # clusters are empty
  q[which(q<1e-60)] = 1e-60
  q[which(q>1-1e-60)] = 1e-60
  q = q/matrix(rep(rowSums(q),K),ncol=K,byrow=FALSE)
  # Update priors
  priors = colMeans(q)
  # Update means
  for(k in 1:K){
    means[k,] = colSums(X*matrix(rep(q[,k],D),ncol=D,byrow=FALSE))/sum(q[,k])
  }
  # update covariances
  for(k in 1:K){
    Xm = X - matrix(rep(means[k,],N),nrow=N,byrow=TRUE)
    covs[[k]] = t(Xm*matrix(rep(q[,k],D),ncol=D,byrow=FALSE))%*%Xm
    covs[[k]] = covs[[k]]/sum(q[,k])
  }
  
  ## Plot the current status
  if(any(it==plotpoints)){
    # Note the following plots points using q as their RGB colour value
    plot(X[1,1],X[1,2],col=rgb(q[1,1],q[1,2],q[1,3]),pch=16,xlim=c(-6,6),ylim=c(-6,6),main=paste("After iteration",it))
    for(n in 1:N){
      points(X[n,1],X[n,2],pch=16,col=rgb(q[n,1],q[n,2],q[n,3]))
    }
    gridvals = seq(-6,6,0.1)
    mesh = meshgrid(gridvals,gridvals)
    Xv = mesh$X; Yv = mesh$Y
    grid = cbind(as.vector(Xv),as.vector(Yv))
    for(k in 1:K){
      Probs = dmvnorm(x=grid,mean=t(means[k,]),sigma=covs[[k]])
      Probs = matrix(Probs,nrow=dim(Xv)[1],ncol=dim(Xv)[2],byrow=TRUE)
      contour(gridvals,gridvals,Probs,drawlabels = FALSE,add=TRUE)
    }
  }    
}

## Plot the bound
plot(2:length(B),B[2:length(B)],xlab="Iterations",ylab="Bound",type="l")

