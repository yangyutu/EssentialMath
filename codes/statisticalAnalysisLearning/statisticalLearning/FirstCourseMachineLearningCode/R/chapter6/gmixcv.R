## gmixcv.R
# From A First Course in Machine Learning, Chapter 6.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
# CV for selection of K in a Gaussian Mixture
rm(list=ls(all=TRUE))

## Load the data
setwd("~/data/kmeansdata") ## Might need to change the path here
X = as.matrix(read.csv(file="X.csv",header=FALSE))

## Plot the data
plot(X[,1],X[,2],pch=16,xlim=c(-6,6),ylim=c(-6,6))

## Do a 10-fold CV
NFold = 10
N = nrow(X)
sizes = rep(floor(N/NFold),NFold)
sizes[NFold] = sizes[NFold] + N - sum(sizes)
csizes = c(0,cumsum(sizes))
order = sample.int(N) # Randomise the data order
MaxIts = 100
## Loop over K
Kvals = 1:10
outlike = matrix(NA,nrow=length(Kvals),ncol=N)

for(kv in 1:length(Kvals)){
  K = Kvals[kv]
  for(fold in 1:NFold){
    cat('\nK:',K,'fold:',fold)
    foldindex = order[(csizes[fold]+1):csizes[fold+1]]
    trainX = X[-foldindex,] 
    testX = X[foldindex,]
  
    # Train the mixture
    means = matrix(rnorm(K*2),nrow=K,ncol=2)
    covs = list()
    for(k in 1:K){
      covs[[k]] = runif(1)*diag(2)+1*(10^0)*diag(2)
    }
    priors = rep(1/K,K)
    B = -Inf;
    converged = FALSE
    it = 0
    tol = 1e-2
    Ntrain = nrow(trainX)
    D = ncol(X)
    temp = matrix(0,nrow=Ntrain,ncol=K)
    
    
    while(converged == FALSE && it<=MaxIts){
      it = it + 1
      # Update q
      for(k in 1:K){
        const = -(D/2)*log(2*pi) - 0.5*log(det(covs[[k]]))
        Xm = trainX - matrix(rep(means[k,],Ntrain),nrow=Ntrain,byrow=TRUE)
        temp[,k] = const - 0.5*diag(Xm%*%solve(covs[[k]])%*%t(Xm))
      }
      
      # Compute the Bound on the likelihood
      if(it>1){
        B = c(B,sum(q*log(matrix(rep(priors,Ntrain),nrow=Ntrain,byrow=TRUE))))
        B[it] = B[it] + sum(q*temp) - sum(q*log(q))
        if(abs(B[it]-B[it-1])<tol){
          converged = TRUE
        }
      }
      temp = temp + matrix(rep(priors,Ntrain),nrow=Ntrain,byrow=TRUE)
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
        means[k,] = colSums(trainX*matrix(rep(q[,k],D),ncol=D,byrow=FALSE))/sum(q[,k])
      }
      # update covariances
      for(k in 1:K){
        Xm = trainX - matrix(rep(means[k,],Ntrain),nrow=Ntrain,byrow=TRUE)
        covs[[k]] = t(Xm*matrix(rep(q[,k],D),ncol=D,byrow=FALSE))%*%Xm
        covs[[k]] = covs[[k]]/sum(q[,k])
      }
    }
    # Compute the held-out likelihood
    Ntest = nrow(testX)
    temp = matrix(0,nrow=Ntest,ncol=K)
    for(k in 1:K){
      const = -(D/2)*log(2*pi) - 0.5*log(det(covs[[k]]))
      Xm = testX - matrix(rep(means[k,],Ntest),nrow=Ntest,byrow=TRUE)
      temp[,k] = const - 0.5 * diag(Xm%*%solve(covs[[k]])%*%t(Xm))
    }
    temp = exp(temp) + matrix(rep(priors,Ntest),nrow=Ntest,byrow=TRUE)
    outlike[kv,(csizes[fold]+1):csizes[fold+1]] = as.vector(log(rowSums(temp)))    
  }
}

## Plot the evolution in log likelihood
plot(Kvals,rowMeans(outlike),type="l",ylim=c(0,0.3),
     xlab="K",ylab="Average held out likelihood")
points(Kvals,rowMeans(outlike)-2*apply(X=outlike,MARGIN=1,FUN=std)/sqrt(N),type="l",col="red")
points(Kvals,rowMeans(outlike)+2*apply(X=outlike,MARGIN=1,FUN=std)/sqrt(N),type="l",col="red")
