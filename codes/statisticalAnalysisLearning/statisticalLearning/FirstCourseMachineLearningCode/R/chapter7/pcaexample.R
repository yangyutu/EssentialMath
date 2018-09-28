## pcaexample.R
# From A First Course in Machine Learning, Chapter 7.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
# PCA example
rm(list=ls(all=TRUE))

## Generate the data
Y = matrix(rnorm(60*2),nrow=60,ncol=2)
Y[21:40,1:2] = Y[21:40,1:2]+5
Y[41:60,1:2] = Y[41:60,1:2]-5
# Add 5 random dimensions
N = nrow(Y)
Y = cbind(Y,matrix(runif(N*5),nrow=N,ncol=5))
# labels - just used for plotting
t = rep(1:3,each=20)

## Plot the original data
symbs = c(1,2,4)
cols = c("red","blue","darkgreen")
pos = which(t==1)
plot(Y[pos,1],Y[pos,2],pch=symbs[1],col=cols[1],xlim=c(-10,10),ylim=c(-10,10))
for(k in 2:3){
  pos = which(t==k)
  points(Y[pos,1],Y[pos,2],pch=symbs[k],col=cols[k])
}

## Do the PCA
# Subtract the means
Y = Y - matrix(rep(colMeans(Y),N),nrow=N,byrow=TRUE)
# Compute covariance matrix
C = (1/N)*t(Y)%*%Y
# Find the eigen-vectors/values
# columns of w correspond to the projection directions
eig = eigen(C)
w =  eig$vectors
lam = eig$values

## Plot the first two components on to the original data
pos = which(t==1)
plot(Y[pos,1],Y[pos,2],pch=symbs[1],col=cols[1],xlim=c(-10,10),ylim=c(-10,10))
for(k in 2:3){
  pos = which(t==k)
  points(Y[pos,1],Y[pos,2],pch=symbs[k],col=cols[k])
}
xl = c(-10,10)
for(k in 1:2){
  points(xl,xl*w[1,k]/w[2,k],type="l")
}

## Bar plot of the eigenvalues
barplot(diag(lam),xlab="Projection dimension",ylab="Variance")

## Plot the data projected into the first two dimensions
X = Y%*%w[,1:2]
pos = which(t==1)
plot(X[pos,1],X[pos,2],pch=symbs[1],col=cols[1],xlim=c(-10,10),ylim=c(-10,10))
for(k in 2:3){
  pos = which(t==k)
  points(X[pos,1],X[pos,2],pch=symbs[k],col=cols[k])
}

