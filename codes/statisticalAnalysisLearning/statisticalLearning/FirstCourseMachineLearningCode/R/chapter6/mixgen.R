## mixgen.R
# From A First Course in Machine Learning, Chapter 6.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
# Generating data from a mixture
rm(list=ls(all=TRUE))
require(mvtnorm) # might require to install this package: install.packages("mvtnorm")
require(pracma) # might require to install this package: install.packages("pracma")

## Define the mixture components
mixture_means = matrix(c(3,3,1,-3),ncol=2,byrow=TRUE)
mixture_covs = list()
mixture_covs[[1]] = matrix(c(1,0,0,2),ncol=2,byrow=TRUE)
mixture_covs[[2]] = matrix(c(2,0,0,1),ncol=2,byrow=TRUE)
priors = c(0.7,0.3)

## Generate data points one at a time
plotpoints = c(1:5,seq(10,30,5),40,50,100,200)
X = matrix(NA,nrow=200,ncol=2)

for(n in 1:200){
  ##
  # Flip a biased coin to choose from the prior
  comp = which(runif(1)<cumsum(priors))
  comp = comp[1]
  X[n,] = rmvnorm(n=1,mean=t(mixture_means[comp,]),sigma=mixture_covs[[comp]])
  if(any(plotpoints==n)){
    plot(X[1:n,1],X[1:n,2],xlim=c(-7,7),ylim=c(-7,7),pch=16,main=paste("N=",n))
    # Make the contours
    gridvals = seq(-7,7,0.1)
    mesh = meshgrid(gridvals,gridvals)
    Xv = mesh$X; Yv = mesh$Y
    grid = cbind(as.vector(Xv),as.vector(Yv))
    for(k in 1:2){
      Probs = dmvnorm(x=grid,mean=t(mixture_means[k,]),sigma=mixture_covs[[k]])
      Probs = matrix(Probs,nrow=dim(Xv)[1],ncol=dim(Xv)[2],byrow=TRUE)
      contour(gridvals,gridvals,Probs,drawlabels = FALSE,add=TRUE)
    }
  }
}
