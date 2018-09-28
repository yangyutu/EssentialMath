## plotcc.R
# From A First Course in Machine Learning, Chapter 5.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
# Computing and plotting class-conditional densities for a Bayes classifier
rm(list=ls(all=TRUE))
require(pracma) # might require to install this package: install.packages("pracma")

## Load the data
setwd("~/data/bc_data") ## Might need to change the path here
X = as.matrix(read.csv(file="X.csv",header=FALSE))
t = as.matrix(read.csv(file="t.csv",header=FALSE))

# Plot the data
cl = as.vector(unique(t))
col = c('red','green','blue')
pos = which(t==cl[1])
plot(X[pos,1],X[pos,2],xlim=c(-3,7),ylim=c(-6,6),pch=16,col=col[1])
for(c in 2:length(cl)){
  pos = which(t==cl[c])
  points(X[pos,1],X[pos,2],col=col[c],pch=16)
}

## Fit class-conditional Gaussians for each class
# Using the Naive (independence) assumption
class_mean = matrix(NA,ncol=dim(X)[2],nrow=length(cl))
class_var = matrix(NA,ncol=dim(X)[2],nrow=length(cl))
for(c in 1:length(cl)){
  pos = which(t==cl[c])
  # Find the means
  class_mean[c,] = apply(X=X[pos,],MARGIN=2,FUN=mean)
  class_var[c,] = apply(X=X[pos,],MARGIN=2,FUN=var)
}

## Plot the contours
gridvalsX = seq(-6,6,0.1)
gridvalsY = seq(-6,6,0.1)
mesh = meshgrid(gridvalsX,gridvalsY)
Xv = mesh$X; Yv = mesh$Y

for(c in 1:length(cl)){
  temp = cbind(as.vector(Xv)-class_mean[c,1], as.vector(Yv)-class_mean[c,2])
  tempc = diag(class_var[c,])
  const = -log(2*pi) - log(det(tempc))
  Probs = exp(const - (0.5*diag(temp%*%solve(tempc)%*%t(temp))))
  contour(gridvalsX,gridvalsY,matrix(Probs,nrow=dim(Xv)[[1]],ncol=dim(Xv)[[2]],byrow=TRUE),add=TRUE)
}

## Repeat without Naive assumption
class_var = list()
for(c in 1:length(cl)){
  pos = which(t==cl[c])
  # Find the means
  class_mean[c,] = apply(X=X[pos,],MARGIN=2,FUN=mean)
  class_var[[c]] = cov(X[pos,])
}

## Plot the contours
cl = as.vector(unique(t))
col = c('red','green','blue')
pos = which(t==cl[1])
plot(X[pos,1],X[pos,2],xlim=c(-3,7),ylim=c(-6,6),pch=16,col=col[1])
for(c in 2:length(cl)){
  pos = which(t==cl[c])
  points(X[pos,1],X[pos,2],col=col[c],pch=16)
}

gridvalsX = seq(-6,6,0.1)
gridvalsY = seq(-6,6,0.1)
mesh = meshgrid(gridvalsX,gridvalsY)
Xv = mesh$X; Yv = mesh$Y

for(c in 1:length(cl)){
  temp = cbind(as.vector(Xv)-class_mean[c,1], as.vector(Yv)-class_mean[c,2])
  tempc = class_var[[c]]
  const = -log(2*pi) - log(det(tempc))
  Probs = exp(const - (0.5*diag(temp%*%solve(tempc)%*%t(temp))))
  contour(gridvalsX,gridvalsY,matrix(Probs,nrow=dim(Xv)[[1]],ncol=dim(Xv)[[2]],byrow=TRUE),add=TRUE)
}

