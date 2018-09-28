## mhexample.R
# From A First Course in Machine Learning, Chapter 4.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
# Example of Metropolis-Hastings
rm(list=ls(all=TRUE))

## Define the true density as a Gaussian
mu = c(2,3)
si = matrix(c(1,0.6,0.6,1),ncol=2)

## Initialise sampler
x = c(4,0.5)

## Do N steps
gridvals = seq(-2,6,0.1)
mesh = meshgrid(gridvals,gridvals)
Xv = mesh$X; Yv=mesh$Y
const = -log(2*pi) - log(det(si))
temp = rbind(as.vector(Xv)-mu[1],as.vector(Yv)-mu[2])
Probs = const - 0.5*diag(t(temp)%*%solve(si)%*%temp)
plot(x[1],x[2],xlim=c(-1,6),ylim=c(-1,6))
contour(gridvals,gridvals,matrix(exp(Probs),ncol=dim(Xv)[1],nrow=dim(Xv)[2],byrow=TRUE),drawlabels=FALSE,add=TRUE)


N = 40 # Increase this to generate more samples
jump_si = matrix(c(0.5,0,0,0.5),ncol=2) # Covariance of jumping Gaussian - try varying this and looking at the proportion of rejections/acceptances
Naccept = 0

x = rbind(x,matrix(NA,ncol=2,nrow=N))
for(n in 2:(N+1)){
  xs = mvrnorm(1,x[n-1,],jump_si) #Using a Gaussian jump, jump ratios cancel
  # Compute ratio of densities (done in log space, constants cancel)
  pnew = -0.5*t(xs-mu)%*%solve(si)%*%(xs-mu)
  pold = -0.5*t(x[n-1,]-mu)%*%solve(si)%*%(x[n-1,]-mu)
  if(runif(1) <= exp(pnew-pold)){
    # Accept the sample
    x[n,] = xs
    points(x[n,1],x[n,2],pch=16)
    points(x[1:n,1],x[1:n,2],type="l")
  }
  else{
    x[n,] = x[n-1,]
    points(xs[1],xs[2],pch=16)
    points(c(x[n-1,1],xs[1]),c(x[n-1,2],xs[2]),type="l",col="red")
  }
}
