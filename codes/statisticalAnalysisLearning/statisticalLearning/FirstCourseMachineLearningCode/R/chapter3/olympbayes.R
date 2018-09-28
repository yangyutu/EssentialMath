## olympbayes.R
# From A First Course in Machine Learning, Chapter 3.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
# Bayesian treatment of Olympic data
rm(list=ls(all=TRUE))
require(MASS) # might require to install this package: install.packages("MASS")
require(pracma) # might require to install this package: install.packages("pracma")

## Load Olympic data
setwd("~/data/olympics") ## Might need to change the path here
male100 = read.csv(file="male100.csv",header=FALSE)

x = male100[,1]
t = male100[,2]

# Rescale x for numerical stability
x = x - x[1]
x = x/30

## Define the prior
# $p(\mathbf{w}) = {\cal N}(\mu_0,\Sigma_0)
mu0 = matrix(c(0,0),ncol=1)
si0 = matrix(c(100,0,0,5),ncol=2)
ss = 2 # Vary this to see the effect on the posterior samples

## Draw some functions from the prior
w = mvrnorm(n=10,mu0,si0)
X = cbind(x^0,x^1)

# Plot the data and the function
plot(x,t,ylim=c(0,20))
for(i in 1:10){
  y = X%*%as.matrix(w[i,])
  lines(x,y,lty=2,col="darkgreen")
}

## Add the data 3 points at a time
dord = seq(3,length(x),3)
for(i in 1:length(dord)){
  ##
  Xsub = X[1:dord[i],]
  tsub = t[1:dord[i]]
  siw = solve((1/ss)*t(Xsub)%*%Xsub + solve(si0))
  muw = siw%*%((1/ss)*t(Xsub)%*%tsub + solve(si0)%*%mu0)
  
  par(mar = c(5, 4, 4, 12))
  plot(x,t,ylim=c(5,15),xlab="Olympic number",ylab="Winning time")
  points(x,X%*%muw,type="l",col="blue")
  wsamp = mvrnorm(n=10,muw,siw)
  for(j in 1:10){
    lines(x,t(X%*%wsamp[j,]),lty=2,col="darkgreen")
  }
  par(xpd=T)
  legend(x=4.5,y=15,legend=c('Data','Posterior mean','Samples'),
         pch=c(1,NA,NA),lty=c(NA,1,2),col=c("black","blue","darkgreen"),merge=TRUE)
  par(xpd=F,mar = c(5, 4, 4, 2))
  
  # Contour plot the prior and posterior
  gridvalsx = seq(9,13,0.05); gridvalsy= seq(-1.5,0.5,0.05)
  mesh = meshgrid(gridvalsx,gridvalsy)
  Xv = mesh$X; Yv=mesh$Y
  
  const = (1/sqrt(2*pi))^2
  const_prior = const/sqrt(det(si0))
  const = const/sqrt(det(siw))
  
  temp = cbind(as.matrix(c(Xv))-muw[1],as.matrix(c(Yv))-muw[2])
  temp_prior = cbind(as.matrix(c(Xv))-mu0[1],as.matrix(c(Yv))-mu0[2])
  pdfv = const*exp(-0.5*diag(temp%*%solve(siw)%*%t(temp)))
  pdfv = matrix(pdfv,nrow=dim(Xv)[1],ncol=dim(Xv)[2],byrow=F)
  pdfv_prior = const*exp(-0.5*diag(temp_prior%*%solve(si0)%*%t(temp_prior)))
  pdfv_prior = matrix(pdfv_prior,nrow=dim(Xv)[1],ncol=dim(Xv)[2],byrow=F)
  
  contour(x=gridvalsx,y=gridvalsy,z=t(pdfv),drawlabels=FALSE)
  contour(x=gridvalsx,y=gridvalsy,z=t(pdfv_prior),drawlabels=FALSE,add=TRUE,col="red")      
}





