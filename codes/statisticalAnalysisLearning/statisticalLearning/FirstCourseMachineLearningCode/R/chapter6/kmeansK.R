## kmeansK.R
# From A First Course in Machine Learning, Chapter 6.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
# Overfitting in K-means
rm(list=ls(all=TRUE))

## Load the data
setwd("~/data/kmeansdata") ## Might need to change the path here
X = as.matrix(read.csv(file="X.csv",header=FALSE))

## Plot the data
plot(X[,1],X[,2],pch=16,xlim=c(-3,5),ylim=c(-7,7))

Kvals = 1:10
Nreps = 50
total_distance = matrix(0,nrow=length(Kvals),ncol=Nreps)

for(kv in 1:length(Kvals)){
  for(rep in 1:Nreps){
    K = Kvals[kv]
    cluster_means = matrix(runif(K*2),nrow=K)*10-5
    converged = FALSE
    N = nrow(X)
    cluster_assignments = matrix(0,nrow=N,ncol=K)
    di = matrix(0,nrow=N,ncol=K)
    
    while(!converged){
    
      # Update assignments
      for(k in 1:K){
        di[,k] = rowSums((X - matrix(rep(cluster_means[k,],N),ncol=2,byrow=TRUE))^2)
      }
      old_assignments = cluster_assignments
      cluster_assignments = (di == matrix(rep(apply(X=di,MARGIN=1,FUN=min),K),ncol=K,byrow=FALSE))
      if(sum(sum(old_assignments!=cluster_assignments))==0){
        converged = TRUE
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
    }
    # Compute the distance
    total_distance[kv,rep] = sum(di*cluster_assignments) 
  }
}

## Make the boxplot
boxplot(t(log(total_distance)),xlab="K",ylab="Log D")


