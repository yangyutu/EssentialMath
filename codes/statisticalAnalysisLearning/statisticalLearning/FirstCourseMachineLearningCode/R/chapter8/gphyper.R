## gphyper.R
# Performs binary GP classification with two-dimensional data
#
# From A First Course in Machine Learning
# Francois-Xavier Briol, 31/10/16 [f-x.briol@warwick.ac.uk]
rm(list=ls(all=TRUE))

## Generate the data set, and set the GP hyperparameters
ss = 0.05
alpha = 1.0
gamma = 5.0
N = 10
x = runif(N)
x = matrix(sort(x),nrow=N)

## Compute the covariance function
Xdi = (matrix(rep(x,nrow(x)),nrow=nrow(x))-matrix(rep(x,nrow(x)),nrow=nrow(x),byrow=T))^2
C = alpha*exp(-gamma*Xdi);
  
## Sample a true latent function
f = gausssamp(matrix(0,nrow=N,ncol=1),C,1)
  
# Plot the function
plot(x,f,xlab="x",ylab="y",pch=16,xlim=c(0,1),ylim=c(-2,2))
f = t(f)
y = f + rnorm(N)*sqrt(ss)
points(x,y,pch=16,col="red")

## Vary gamma and ss (noise variance) on a grid and see what the marginal likelihood looks like
gridseqx = seq(from=-5,to=5,by=0.1)
gridseqy = seq(from=-5,to=0,by=0.1)
mesh=meshgrid(gridseqx,gridseqy)
G = mesh$X; SS = mesh$Y
ML = matrix(0,ncol=ncol(G),nrow=nrow(G))
for(i in 1:prod(dim(G))){
  CSS = alpha*exp(-exp(G[i])*Xdi) + exp(SS[i])*diag(N)
  ML[i] = -(N/2)*log(2*pi) - 0.5*log(det(CSS)) - 0.5*t(y)%*%solve(CSS)%*%y
}

# Contour the results
filled.contour(gridseqx,gridseqy,t(ML),xlab="",ylab="x_2",plot.axes={points(log(gamma),log(ss),pch=16)})
  