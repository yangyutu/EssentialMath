## loglap.R
# From A First Course in Machine Learning, Chapter 4.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
# The Laplace approximation for logistic regression
rm(list=ls(all=TRUE))
require(pracma) # might require to install this package: install.packages("pracma")
require(MASS) # might require to install this package: install.packages("MASS")

setwd("~/data/logregdata") ## Might need to change the path here

X = as.matrix(read.csv(file="X.csv",header=FALSE))
t = as.matrix(read.csv(file="t.csv",header=FALSE))

## Find the mode and the Hessian (see logmap.R)
w = matrix(c(0,0),ncol=1) # Start at zero
tol = 1e-6 # Stopping tolerance
Nits = 100
w_all = matrix(0,nrow=Nits,ncol=2) # Store evolution of w values
ss = 10 # Prior variance on the parameters of w
change = Inf
it = 0
while(change>tol && it<=100){
  prob_t = 1/(1+exp(-X%*%w))
  # Gradient
  grad = -(1/ss)*t(w) 
  part1 = do.call(cbind, replicate(length(w), t, simplify=FALSE))
  part2 = do.call(cbind, replicate(length(w), prob_t, simplify=FALSE))
  grad = grad + colSums(X*(part1-part2))
  # Hessian
  H = -t(X)%*%(diag(as.vector(prob_t*(1-prob_t))))%*%X
  H = H - (1/ss)*diag(length(w))
  # Update w
  w = w - solve(H)%*%t(grad)
  it = it + 1
  w_all[it,] = t(w)
  if(it>1){
    change = sum((w_all[it,] - w_all[it-1,])^2)
  }
}
w_all = w_all[-((it+1):100),] 

## Set the Laplace approximation
muw = w
siw = solve(-H)

## Plot the true posterior (note that we can only get this in unnormalised form)
gridvals = seq(-5,5,0.1)
mesh = meshgrid(gridvals,gridvals)
w1 = mesh$X; w2=mesh$Y
logprior = -0.5*log(2*pi) - 0.5*log(ss) - (1/(2*ss))*(w1^2)
logprior = logprior + (-0.5*log(2*pi) - 0.5*log(ss) - (1/(2*ss))*(w2^2))
prob_t = 1/(1+exp(-cbind(as.vector(w1),as.vector(w2))%*%t(X)))
loglike = rowSums(log(prob_t)*t(do.call(cbind, replicate(prod(size(w1)),t, simplify=FALSE))))
loglike = loglike + rowSums(log(1-prob_t)*t(do.call(cbind, replicate(prod(size(w1)),1-t, simplify=FALSE))))
logpost = logprior + matrix(loglike,ncol=dim(w1)[1],nrow=dim(w1)[2])
contour(gridvals,gridvals,exp(logpost),xlim=c(-5,5),ylim=c(-5,5),xlab="w1",ylab="w2",drawlabels=FALSE)                                             

## Overlay the approximation
temp = cbind(as.vector(w1)-muw[1,],as.vector(w2)-muw[2,])
D = 2; # Working in 2 dimensions
logconst = -(D/2)*log(2*pi) - 0.5*log(det(siw))
log_truepost = logconst - diag(0.5*temp%*%solve(siw)%*%t(temp))
contour(gridvals,gridvals,matrix(exp(log_truepost),ncol=dim(w1)[1],nrow=dim(w1)[2]),add=TRUE,col="red",drawlabels=FALSE)
legend(x=-5,y=-1,legend=c('True','Laplace'),lty=c(1,1),col=c("black","red"))
                               
## Plot the decision contours

# Create an x grid
gridvals = seq(-5,5,0.1)
mesh = meshgrid(gridvals,gridvals)
Xv = mesh$X; Yv=mesh$Y

# Generate samples from the approximate posterior
Nsamps = 1000
w_samps = mvrnorm(n=Nsamps,mu=muw,Sigma=siw)


# Compute the probabilities over the grid by averaging over the samples
Probs = matrix(0,ncol=dim(Xv)[1],nrow=dim(Xv)[2])
for(i in 1:Nsamps){
  Probs = Probs + 1/(1 + exp(-(w_samps[i,1]*Xv + w_samps[i,2]*Yv)))
}
Probs = Probs/Nsamps
plot(X[1:20,1],X[1:20,2],xlim=c(min(X),max(X)),ylim=c(min(X),max(X)),xlab="X[,1]",ylab="X[,2]",pch=16)
points(X[21:40,1],X[21:40,2],pch=0)
contour(x=gridvals,y=gridvals,z=Probs,drawlabels=FALSE,add=TRUE)

