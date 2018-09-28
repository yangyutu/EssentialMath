## svmgauss.R
# From A First Course in Machine Learning, Chapter 5.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
# SVM with Gaussian kernel
rm(list=ls(all=TRUE))
require(quadprog) # might require to install this package: install.packages("quadprog")
require(pracma) # might require to install this package: install.packages("pracma")

## Load the data
setwd("~/data/SVMdata2") ## Might need to change the path here

X = as.matrix(read.csv(file="X.csv",header=FALSE))
t = as.matrix(read.csv(file="t.csv",header=FALSE))

# Put in class order for visualising the kernel
I = order(t)
t = sort(t);
X = X[I,]

## Plot the data
tv = as.vector(unique(t))
ma = c(0,16)
col= c("blue","red")
pos = which(t==tv[1])
plot(X[pos,1],X[pos,2],col=col[1],pch=ma[1],xlim=c(-4,4),ylim=c(-4,4))
for(i in 1:length(tv)){
  pos = which(t==tv[i])
  points(X[pos,1],X[pos,2],col=col[i],pch=ma[i])
}

## Compute Kernel and test Kernel
gridvals = seq(-4,4,0.1)
mesh = meshgrid(gridvals,gridvals)
Xv = mesh$X; Yv = mesh$Y
testX = cbind(as.vector(Xv),as.vector(Yv))
N =nrow(X)
Nt = nrow(testX)
K = matrix(0,nrow=N,ncol=N)
testK = matrix(0,nrow=N,ncol=Nt)

# Set kernel parameter
gamvals = c(0.01,1,5,10,50)

for(gv in 1:length(gamvals)){
  gam = gamvals[gv]
  for(n in 1:N){
    for(n2 in 1:N){
      K[n,n2] = exp(-gam*sum((X[n,]-X[n2,])^2))
    }
    for(n2 in 1:Nt){
      testK[n,n2] = exp(-gam*sum((X[n,]-testX[n2,])^2))
    }
  }
  image(t(K),main=paste("Gamma=",gam))
  
  ## Setup the SVM optimisation problem in the matrix form below
  # $$ \min \frac{1}{2}b^T D b - d^T b$$
  # s.t. $$ A^T b \geq b_0 $
  # where the first inequality is treated as an equality.
  D = (t%*%t(t))*K + (1e-5)*diag(N)
  d = matrix(rep(1,N))
  
  # Fix C
  C = 10
  A = cbind(t,diag(N),(-1)*diag(N))
  b0 = c(rep(0,N+1),rep(-C,N))
  
  # Following line runs the SVM optimisation
  alpha = solve.QP(Dmat=D,dvec=d,Amat=A,bvec=b0,meq=1)
  
  # Compute the bias
  fout = colSums(matrix(alpha$solution*t,ncol=N,nrow=N,byrow=FALSE)*K)
  pos = which(alpha$solution>1e-3)
  bias = mean(t[pos]-fout[pos])
  
  # Compute the test predictions
  testpred = t(alpha$solution*t)%*%testK + bias
  testpred = t(testpred)
  
  # Plot the data, decision boundary and Support vectors
  tv = as.vector(unique(t))
  ma = c(0,16)
  col= c("blue","red")
  pos = which(t==tv[1])
  plot(X[pos,1],X[pos,2],col=col[1],pch=ma[1],xlim=c(-4,4),ylim=c(-4,4),
       main=paste("Gamma=",gam))
  for(i in 1:length(tv)){
    pos = which(t==tv[i])
    points(X[pos,1],X[pos,2],col=col[i],pch=ma[i])
  }
  contour(gridvals,gridvals,matrix(testpred,
          nrow=dim(Xv)[1],ncol=dim(Xv)[2],byrow=TRUE),add=TRUE)
}