## gauss_surf.R
# From A First Course in Machine Learning, Chapter 2.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
# Surface and contour plots of a Gaussian
rm(list=ls(all=TRUE))
require(pracma) # might require to install this package: install.packages("pracma")
require(rgl) # might require to install this package: install.packages("rgl")

## The Multi-variate Gaussian pdf is given by:
# $p(\mathbf{x}|\mu,\Sigma) =
# \frac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}}\exp\left\{-\frac{1}{2}(\mathbf{x}-\mu)^T\Sigma^{-1}(\mathbf{x}-\mu)\right\}$
## Define the Gaussian
mu = matrix(c(1,2),ncol=1)
sigma = matrix(c(1,0.8,0.8,4),ncol=2)

## Define the grid for visualisation
gridvals = seq(-5,5,0.1)
mesh = meshgrid(gridvals,gridvals)
X = mesh$X; Y=mesh$Y

## Define the constant
const = (1/sqrt(2*pi))^2
const = const/sqrt(det(sigma))
temp = cbind(as.matrix(c(X))-mu[1],as.matrix(c(Y))-mu[2])
pdfv = const*exp(-0.5*diag(temp%*%solve(sigma)%*%t(temp)))
pdfv = matrix(pdfv,nrow=dim(X)[1],ncol=dim(X)[2],byrow=T)

## Make the plots
par(mfrow=c(1,2))
contour(x=gridvals,y=gridvals,z=pdfv,drawlabels=FALSE)
persp(x = gridvals, y= gridvals, z = pdfv)
par(mfrow=c(1,1))


