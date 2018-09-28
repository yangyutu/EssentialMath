## kmeansexample.R
# From A First Course in Machine Learning, Chapter 6.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
# Example of K-means clustering
rm(list=ls(all=TRUE))

## Load the data
setwd("~/data/kmeansdata") ## Might need to change the path here
X = as.matrix(read.csv(file="X.csv",header=FALSE))

## Plot the data
plot(X[,1],X[,2],pch=16,xlim=c(-3,5),ylim=c(-7,7))

## Randomly initialise the means
K = 3 # The number of clusters
cluster_means = matrix(runif(K*2),nrow=K)*10-5

## Iteratively update the means and assignments
converged = FALSE
N = nrow(X)
cluster_assignments = matrix(0,nrow=N,ncol=K)
di = matrix(0,nrow=N,ncol=K)
cols = c("red","darkgreen","blue")

while(!converged){
  ##
  # Update assignments
  for(k in 1:K){
    di[,k] = rowSums((X - matrix(rep(cluster_means[k,],N),ncol=2,byrow=TRUE))^2)
  }
  old_assignments = cluster_assignments
  cluster_assignments = (di == matrix(rep(apply(X=di,MARGIN=1,FUN=min),K),ncol=K,byrow=FALSE))
  if(sum(sum(old_assignments!=cluster_assignments))==0){
    converged = TRUE
  }
  
  # Plot the assigned data
  plot(X[cluster_assignments[,1],1],X[cluster_assignments[,1],2],
       col=cols[1],pch=16,main="Updated Assignments",xlim=c(-3,5),ylim=c(-7,7))
  for(k in 1:K){
    points(X[cluster_assignments[,k],1],X[cluster_assignments[,k],2],
    col=cols[k],pch=16)
  }
  
  # Update means
  for(k in 1:K){ 
    if(sum(cluster_assignments[,k])==0){
      # This cluster is empty, randomise it
      cluster_means[k,] = runif(2)*10-5
    } else {
      current_assign = X[cluster_assignments[,k],]
      current_assign = matrix(current_assign,ncol=ncol(X))
      cluster_means[k,] = colMeans(current_assign)
    }
  }
  # Plot the means
  for(k in 1:K){ 
    points(cluster_means[k,1],cluster_means[k,2],pch=8,col=cols[k],cex=2)
  }
}



