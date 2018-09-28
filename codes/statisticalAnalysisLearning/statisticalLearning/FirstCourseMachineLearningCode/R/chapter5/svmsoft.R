## svmsoft.R
# From A First Course in Machine Learning, Chapter 5.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
# Soft margin SVM
rm(list=ls(all=TRUE))
require(quadprog) # might require to install this package: install.packages("quadprog")

## Generate the data
x = rbind(matrix(rnorm(2*20),ncol=2),matrix(rnorm(2*20)+4,ncol=2))
t = c(rep(-1,20),rep(1,20))
# Add a bad point
x = rbind(x,c(2,1))
t = c(t,1)

## Plot the data
tv = as.vector(unique(t))
ma = c(0,16)
col= c("blue","red")
pos = which(t==tv[1])
plot(x[pos,1],x[pos,2],col=col[1],pch=ma[1],xlim=c(-3,7),ylim=c(-3,7))
for(i in 1:length(tv)){
  pos = which(t==tv[i])
  points(x[pos,1],x[pos,2],col=col[i],pch=ma[i])
}

## Setup the SVM optimisation problem in the matrix form below
# $$ \min \frac{1}{2}b^T D b - d^T b$$
# s.t. $$ A^T b \geq b_0 $
# where the first inequality is treated as an equality.
N = nrow(x)
K = x%*%t(x)
D = (t%*%t(t))*K + (1e-5)*diag(N)
d = matrix(rep(1,N))


## Loop over various values of the margin parameter
Cvals = c(10,5,2,1,0.5,0.1,0.05,0.01)
for(cv in 1:length(Cvals)){
  ##
  A = cbind(t,diag(N),(-1)*diag(N))
  b0 = c(rep(0,N+1),rep(-Cvals[cv],N))
  
  # Following line runs the SVM
  alpha = solve.QP(Dmat=D,dvec=d,Amat=A,bvec=b0,meq=1)
  
  # Compute the bias
  fout = colSums(matrix(alpha$solution*t,ncol=N,nrow=N,byrow=FALSE)*K)
  pos = which(alpha$solution>1e-3)
  bias = mean(t[pos]-fout[pos])
  
  # Plot the data, decision boundary and Support vectors
  tv = as.vector(unique(t))
  ma = c(0,16)
  col= c("blue","red")
  pos = which(t==tv[1])
  plot(x[pos,1],x[pos,2],col=col[1],pch=ma[1],xlim=c(-3,7),ylim=c(-3,7),
       main=paste("Cval=",Cvals[cv]))
  for(i in 1:length(tv)){
    pos = which(t==tv[i])
    points(x[pos,1],x[pos,2],col=col[i],pch=ma[i])
  }
  
  # Because this is a linear SVM, we can compute w and plot the decision
  # boundary exactly.
  xp = c(-3,7)
  w = colSums(matrix(alpha$solution*t,nrow=N,ncol=ncol(x),byrow=FALSE)*x)
  yp = -(bias + w[1]*xp)/w[2]
  points(xp,yp,type="l")
}