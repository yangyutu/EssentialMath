## logmap.R
# From A First Course in Machine Learning, Chapter 4.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
# Finding the MAP parameter value using logistic regression
rm(list=ls(all=TRUE))
require(pracma) # might require to install this package: install.packages("pracma")

setwd("~/data/logregdata") ## Might need to change the path here
X = as.matrix(read.csv(file="X.csv",header=FALSE))
t = as.matrix(read.csv(file="t.csv",header=FALSE))

# Plot the data
plot(X[1:20,1],X[1:20,2],xlim=c(min(X),max(X)),ylim=c(min(X),max(X)),xlab="X[,1]",ylab="X[,2]",pch=16)
points(X[21:40,1],X[21:40,2],pch=0)

## Initialise the parameters
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

## Plot the evolution of w
plot(1:nrow(w_all),w_all[,1],ylim=range(w_all),type="l",col="blue",xlab="Iterations",ylab="w")
points(1:nrow(w_all),w_all[,2],type="l",col="red")

## Plot the probability contours
plot(X[1:20,1],X[1:20,2],xlim=c(min(X),max(X)),ylim=c(min(X),max(X)),xlab="X[,1]",ylab="X[,2]",pch=16)
points(X[21:40,1],X[21:40,2],pch=0)
gridvals = seq(-5,5,0.1)
mesh = meshgrid(gridvals,gridvals)
Xv = mesh$X; Yv=mesh$Y
Probs = 1/(1+exp(-(w[1,]*Xv+w[2,]*Yv)))
contour(x=gridvals,y=gridvals,z=Probs,drawlabels=FALSE,add=TRUE)
