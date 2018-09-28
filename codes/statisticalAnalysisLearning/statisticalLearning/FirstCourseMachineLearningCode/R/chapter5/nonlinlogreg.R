## nonlinlogreg.R
# From A First Course in Machine Learning, Chapter 5.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
# Logistic regression with nonlinear functions
rm(list=ls(all=TRUE))

## Generate some data
x = matrix(rnorm(100),ncol=2) + cbind(rep(5,50),rep(1,50))
x = rbind(x,matrix(rnorm(100),ncol=2))
x = rbind(x,matrix(rnorm(200),ncol=2)+cbind(rep(2,100),rep(3,100)))
t = matrix(rep(c(0,1),each=100),ncol=1)

## Plot the data
tv = as.vector(unique(t))
ma = c(0,16)
col= c("blue","red")
pos = which(t==tv[1])
plot(x[pos,1],x[pos,2],col=col[1],pch=ma[1],xlim=c(-3,7),ylim=c(-3,7))
for(i in 1:length(tv)){
  pos = which(t==tv[i])
  points(x[pos,1],x[pos,2],col=col[i],pch=ma[i])
}

## Augment the data with nonlinear terms
# $w_0 + w_1x_1 + w_2x_2 + w_3x_1^2 + w_4x_2^2$
X = do.call(cbind, list(x[,1]^0,x,x^2))

## Use the Newton-Raphson MAP solution
w = matrix(rep(0,dim(X)[[2]]),ncol=1) # Start at zero
tol = 1e-6 # Stopping tolerance
Nits = 100
w_all = matrix(0,nrow=Nits,ncol=dim(X)[[2]]) # Store evolution of w values
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
w_all = w_all[-((it+1):nrow(w_all)),]

## Plot the parameter convergence
cols = c("black","blue","red","darkgreen","orange")
plot(1:nrow(w_all),w_all[,1],ylim=range(w_all)+c(-0.5,0.5),type="l",col=cols[1],xlab="Iterations",ylab="Parameters")
for(i in 2:ncol(w_all)){
  points(1:nrow(w_all),w_all[,i],type="l",col=cols[i])
}

## Plot the decision contours
gridvals = seq(from=min(x),to=max(x),by=0.1)
mesh = meshgrid(gridvals,gridvals)
Xv = mesh$X; Yv=mesh$Y
Pvals = 1/(1 + exp(-(w[1,] + w[2,]*Xv + w[3,]*Yv + w[4,]*(Xv^2) + w[5,]*(Yv^2))))
pos = which(t==tv[1])
plot(x[pos,1],x[pos,2],col=col[1],pch=ma[1],xlim=c(-3,7),ylim=c(-3,7))
for(i in 1:length(tv)){
  pos = which(t==tv[i])
  points(x[pos,1],x[pos,2],col=col[i],pch=ma[i])
}
contour(gridvals,gridvals,t(Pvals),add=TRUE)