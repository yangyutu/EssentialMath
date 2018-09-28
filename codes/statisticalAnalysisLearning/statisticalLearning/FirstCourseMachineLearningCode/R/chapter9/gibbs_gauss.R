## gibbs_gauss.R
# Demonstrates use of Gibbs sampling to sample from a multi-dimensional Gaussian
#
# From A First Course in Machine Learning
# Francois-Xavier Briol, 31/10/16 [f-x.briol@warwick.ac.uk]
#
rm(list=ls(all=TRUE))
require(MASS) ## may need install.packages("MASS")
require(pracma) # may need install.packages("pracma")

## First, create the objects to plot the Gaussian contour
# Define the mean and covariance
mu = matrix(c(1,2),ncol=1)
co = matrix(c(1,0.8,0.8,2),byrow=T,ncol=2)

# Define a grid of points for the contours
gridseqx = seq(from=-2,to=4,by=0.1)
gridseqy = seq(from=-1,to=5,by=0.1)
mesh=meshgrid(gridseqx,gridseqy)
Xv = mesh$X; Yv = mesh$Y

# Compute the pdf over the grid
Ci = solve(co)
P = matrix(0,nrow=nrow(Xv),ncol=ncol(Xv))
for(i in 1:nrow(Xv)){
  for(j in 1:nrow(Xv)){
    P[i,j] = -log(2*pi)- 0.5*log(det(co)) - 0.5*(matrix(c(Xv[i,j]-mu[1],Yv[i,j]-mu[2]),ncol=2)%*%Ci%*%matrix(c(Xv[i,j]-mu[1],Yv[i,j]-mu[2]),ncol=1))
  }
}
P = exp(P)


## We now sample with Gibbs sampling using the equations on p.317
##
# Define the initial point - try changing this
x = matrix(c(-1.5,4),ncol=1)
xall = t(x)
yl=c(-1.5,5.5)

# Define when we want to make plots
plot_at = c(1,2,5,10,100)

for(i in 1:100){
  # Sample x_1
  mu_1 = mu[1]+(co[1,2]/co[2,2])*(x[2]-mu[2])
  ss_1 = co[1,1]-co[1,2]^2/co[2,2]
  oldx = x
  x[1] = rnorm(1)*sqrt(ss_1)+mu_1
  xall = rbind(xall,t(x))
         
  # sample x_2
  mu_2 = mu[2]+(co[2,1]/co[1,1])*(x[1]-mu[1])
  ss_2 = co[2,2]-co[2,1]^2/co[1,1]
  oldx = x
  x[2] = rnorm(1)*sqrt(ss_2)+mu_2       
  xall = rbind(xall,t(x))
  
  # If this is a plot iteration, make the plot
  if(any(plot_at==i)){
    filled.contour(gridseqx,gridseqy,t(P),xlab="x_1",ylab="x_2",
          xlim=range(gridseqx),ylim=range(gridseqy),main=paste("After",i,"samples"),
          plot.axes={points(xall[,1],xall[,2],pch=16)})
  }
}
