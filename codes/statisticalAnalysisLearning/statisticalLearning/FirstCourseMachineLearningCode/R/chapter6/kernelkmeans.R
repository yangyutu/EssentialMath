## kernelkmeans.R
# From A First Course in Machine Learning, Chapter 6.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
# Kernel K-means
rm(list=ls(all=TRUE))

## Load the data
setwd("~/data/kmeansnonlindata") ## Might need to change the path here
X = as.matrix(read.csv(file="X.csv",header=FALSE))

## Plot the data
plot(X[,1],X[,2],pch=16,xlim=c(-2,2),ylim=c(-2,2))

## Compute the kernel
N = nrow(X)
Ke = matrix(0,ncol=N,nrow=N)
gam = 1
for(n in 1:N){
  for(n2 in 1:N){
    Ke[n,n2] = exp(-gam*sum((X[n,]-X[n2,])^2))
  }
}

## Run Kernel K-means
converged = FALSE
# Assign all objects into one cluster except one
# Kernel K-means is *very* sensitive to initial conditions.  Try altering
# this initialisation to see the effect.
K = 2 # The number of clusters
Z = matrix(rep(c(1,0),N),nrow=N,byrow=N)
s = rowSums(X^2)
pos = which(s==min(s))
Z[pos,] = c(0,1)
di = matrix(0,nrow=N,ncol=K)
cols = c("red","blue")

## Plot the assignments
for(k in 1:K){
  pos = which(Z[,k]!=0)
  points(X[pos,1],X[pos,2],col=cols[k],pch=16)
}

##
while(!converged){
  Nk = colSums(Z)
  for(k in 1:K){
    # Compute kernelised distance
    di[,k] = diag(Ke) - (2/(Nk[k]))*rowSums(matrix(rep(Z[,k],N),nrow=N,byrow=TRUE)*Ke)
    di[,k] = di[,k] + (Nk[k]^(-2))*sum((Z[,k]%*%t(Z[,k]))*Ke)
  }
  oldZ = Z
  Z = (di == matrix(rep(apply(X=di,MARGIN=1,FUN=min),K),ncol=K,byrow=FALSE))
  if(sum(oldZ!=Z)==0){
    converged = TRUE
  }
  
  ## Plot the assignments
  for(k in 1:K){
    pos = which(Z[,k]!=0)
    points(X[pos,1],X[pos,2],col=cols[k],pch=16)
  }
}
