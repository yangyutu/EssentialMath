## ppcaexample.R
# From A First Course in Machine Learning, Chapter 7.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
# Probabilistic PCA example using Variational Bayes
rm(list=ls(all=TRUE))

## Generate the data
Y = matrix(rnorm(80*2),nrow=80,ncol=2)
Y[21:40,1:2] = Y[21:40,1:2]+5
Y[41:60,1] = Y[41:60,1]+5
Y[61:80,2] = Y[41:60,2]+5

# Add 5 random dimensions
N = nrow(Y)
Y = cbind(Y,matrix(rnorm(N*5),nrow=N,ncol=5))
# labels - just used for plotting
t = rep(1:4,each=20)

## Plot the original data
symbs = c(1,2,3,4)
cols = c("red","blue","darkgreen","orange")
pos = which(t==1)
plot(Y[pos,1],Y[pos,2],pch=symbs[1],col=cols[1],xlim=c(-10,10),ylim=c(-10,10))
for(k in 2:4){
  pos = which(t==k)
  points(Y[pos,1],Y[pos,2],pch=symbs[k],col=cols[k])
}

## Initialise the parameters
a = 1;b = 1; # Hyper-parameters for precision
e_tau = a/b
N = nrow(Y)
M = ncol(Y)
D = 2 # Number of projection dimensions
e_w = matrix(rnorm(M*D),nrow=M,ncol=D)
e_X = matrix(rnorm(N*D),nrow=N,ncol=D)
e_wwt = list()
e_XXt = list()
for(m in 1:M){
  e_wwt[[m]] = diag(D) + t(matrix(e_w[m,],nrow=1))%*%matrix(e_w[m,],nrow=1)
}
tol = 1e-3

## Run the algorithm
MaxIts = 100
B = NA

for(it in 1:MaxIts){
  # Update X
  # Compute \Sigma_x - this is the same for all x
  sigx = solve(diag(D) + as.numeric(e_tau)*Reduce("+",e_wwt))
  for(n in 1:N){
    e_X[n,] = e_tau*sigx%*%matrix(colSums(e_w*t(matrix(rep(Y[n,],D),nrow=D,byrow=TRUE))),ncol=1)
    e_XXt[[n]] = sigx + t(matrix(e_X[n,],nrow=1))%*%matrix(e_X[n,],nrow=1)
  }
  
  # Update W
  sigw = solve(diag(D) + as.numeric(e_tau)*Reduce("+",e_XXt))
  for(m in 1:M){
    e_w[m,] = e_tau*sigw%*%matrix(colSums(e_X*matrix(rep(Y[,m],D),ncol=D,byrow=FALSE)),ncol=1)
    e_wwt[[m]] = sigw +t(matrix(e_w[m,],nrow=1))%*%matrix(e_w[m,],nrow=1)
  }
  
  # Update tau
  e = a + N*M/2;
  # Compute the nasty outer expectation.  Note that these two loops could
  # be made *much* more efficient
  outer_expect = 0
  for(n in 1:N){
    for(m in 1:M){
      outer_expect = outer_expect + sum(diag(e_wwt[[m]]%*%sigx)) + matrix(e_X[n,],nrow=1)%*%e_wwt[[m]]%*%matrix(e_X[n,],ncol=1)
    }
  }
  f = b + 0.5*sum(Y^2) - sum(e_X%*%t(e_w)) + 0.5*outer_expect
  e_tau = as.numeric(e/f)
  e_log_tau = mean(log(rgamma(1000,shape=e,scale=1/f)))
  
  # Compute the bound
  LB = a*log(b) + (a-1)*e_log_tau - b*e_tau - lgamma(a)
  LB = LB - (e*log(f) + (e-1)*e_log_tau - f*e_tau - lgamma(e))
  
  
  for(n in 1:N){
    LB = LB + (-(D/2)*log(2*pi) - 0.5*(sum(diag(sigx))) + matrix(e_X[n,],nrow=1)%*%matrix(e_X[n,],ncol=1))
    LB = LB - (-(D/2)*log(2*pi) - 0.5*log(det(sigx)) - 0.5*D)
  }
  
  for(m in 1:M){
    LB = LB + (-(D/2)*log(2*pi) - 0.5*(sum(diag(sigw))) + matrix(e_X[m,],nrow=1)%*%matrix(e_X[m,],ncol=1))
    LB = LB - (-(D/2)*log(2*pi) - 0.5*log(det(sigw)) - 0.5*D)
  }

  outer_expect = 0
  for(n in 1:N){
    for(m in 1:M){
      outer_expect = outer_expect + sum(diag(e_wwt[[m]]%*%sigx)) + matrix(e_X[n,],nrow=1)%*%e_wwt[[m]]%*%matrix(e_X[n,],ncol=1)
    }
  }
 
  # likelihood bit
  LB = LB + (-(N*M/2)*log(2*pi)+(N*M/2)*e_log_tau-0.5*as.numeric(e_tau)*(sum(Y^2))-2*sum(Y*t((e_w%*%t(e_X)))) + outer_expect)
  B[it] = LB
  
  if(it>2){
    if(abs(B[it]-B[it-1])<tol){
      break
    }
  }
}

## Plot the bound
plot(1:length(B),B,xlab="Iterations",ylab="Bound",type="l")

## Plot the projection
pos = which(t==1)
plot(e_X[pos,1],e_X[pos,2],pch=symbs[1],col=cols[1],xlim=c(-2,2),ylim=c(-2,2),main="Projection")
for(k in 2:4){
  pos = which(t==k)
  points(e_X[pos,1],e_X[pos,2],pch=symbs[k],col=cols[k])
}

