## randwalks.R
# From A First Course in Machine Learning, Chapter 4.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
# Random Walk examples
rm(list=ls(all=TRUE))

## Define the starting point
w = c(0,0)

## Define the jump covariance
si = matrix(c(0.1,0,0,0.1),ncol=2)

## Do N steps
N = 10
w = rbind(w,matrix(NA,ncol=2,nrow=N))
plot(w[1,1],w[1,2],xlim=c(-10,10),ylim=c(-10,10),xlab="w1",ylab="w2")
for(n in 2:(N+1)){
  w[n,] = mvrnorm(1,w[n-1,],si)
  points(w[n,1],w[n,2],pch=0)
  points(w[1:n,1],w[1:n,2],type="l")
}

## Second example
w = c(-2,-2)
si = matrix(c(1,0,0,3),ncol=2)
N = 10
w = rbind(w,matrix(NA,ncol=2,nrow=N))
points(w[1,1],w[1,2],col="red")
for(n in 2:(N+1)){
  w[n,] = mvrnorm(1,w[n-1,],si)
  points(w[n,1],w[n,2],pch=0,col="red")
  points(w[1:n,1],w[1:n,2],type="l",col="red")
}