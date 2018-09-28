## logmh.R
# From A First Course in Machine Learning, Chapter 4.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
# Metropolis-Hastings for Logistic Regression
rm(list=ls(all=TRUE))
require(pracma) # might require to install this package: install.packages("pracma")

## Load the classification data
setwd("~/data/logregdata") ## Might need to change the path here
X = as.matrix(read.csv(file="X.csv",header=FALSE))
t = as.matrix(read.csv(file="t.csv",header=FALSE))

##Initialise w
w = matrix(rnorm(2),ncol=1)

## Generate N Samples
ss = 10
N = 2000
jumpvar = 1 #Jumping variance for each parameter
w_all = matrix(0,nrow=N,ncol=2)
Naccept = 0
for(n in 1:N){
  # Propose a new sample
  ws = w + matrix(rnorm(2),ncol=1)*sqrt(jumpvar)
  # Compute ratio of new to old priors (constants cancel)
  priorrat = -(1/(2*ss))*t(ws)%*%ws
  priorrat = priorrat + (1/(2*ss))*t(w)%*%w # Subtract old prior
  # Compute ratio of new to old likelihoods
  prob = 1/(1+exp(-X%*%w))
  prob = pmax(pmin(prob,1-10^(-12)),10^(-12)) #avoid numerical instability
  newprob = 1/(1+exp(-X%*%ws))
  newprob = pmax(pmin(newprob,1-10^(-12)),10^(-12)) #avoid numerical instability
  like = sum(t*log(prob) + (1-t)*log(1-prob))
  newlike = sum(t*log(newprob) + (1-t)*log(1-newprob))
  rat = newlike - like + priorrat
  if(runif(1) <= exp(as.vector(rat))){
      # Accept
      Naccept = Naccept + 1
      w = ws
  }
  w_all[n,] = w 
}
cat(paste('\nAcceptance ratio:',Naccept/N))


## Plot the true contours and the samples
gridvals = seq(-5,8,0.1)
mesh = meshgrid(gridvals,gridvals)
w1 = mesh$X; w2=mesh$Y
logprior = -0.5*log(2*pi) - 0.5*log(ss) - (1/(2*ss))*(w1^2)
logprior = logprior + (-0.5*log(2*pi) - 0.5*log(ss) - (1/(2*ss))*(w2^2))
prob_t = 1/(1+exp(-cbind(as.vector(w1),as.vector(w2))%*%t(X)))
loglike = rowSums(log(prob_t)*t(do.call(cbind, replicate(prod(size(w1)),t, simplify=FALSE))))
loglike = loglike + rowSums(log(1-prob_t)*t(do.call(cbind, replicate(prod(size(w1)),1-t, simplify=FALSE))))
logpost = logprior + matrix(loglike,ncol=dim(w1)[1],nrow=dim(w1)[2])
contour(gridvals,gridvals,exp(logpost),xlim=c(-5,8),ylim=c(-5,8),xlab="w1",ylab="w2",drawlabels=FALSE)                                             
points(w_all[,1],w_all[,2])

## Plot the prediction contours
# Create an x grid
Xv = mesh$X; Yv=mesh$Y

# Compute the probabilities over the grid by averaging over the samples
Probs = matrix(0,ncol=dim(Xv)[1],nrow=dim(Xv)[2])
for(i in 1:N){
  Probs = Probs + 1/(1 + exp(-(w_all[i,1]*Xv + w_all[i,2]*Yv)))
}
Probs = Probs/N

plot(X[1:20,1],X[1:20,2],xlim=c(min(X),max(X)),ylim=c(min(X),max(X)),xlab="X[,1]",ylab="X[,2]",pch=16)
points(X[21:40,1],X[21:40,2],pch=0)
contour(x=gridvals,y=gridvals,z=Probs,drawlabels=FALSE,add=TRUE)






