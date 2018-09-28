## class2d.R
# From A First Course in Machine Learning, Chapter 8.
# Francois-Xavier Briol, 31/10/16 [f-x.briol@warwick.ac.uk]
# Performs binary GP classification with two-dimensional data
rm(list=ls(all=TRUE))
require(scatterplot3d) # might require install.packages("scatterplot3d")
require(pracma) # might require install.packages("pracma")

## Generate the dataset
# create some random data and then change the means of the two classes to
# separate them
x = matrix(rnorm(40),ncol=2)
x[1:10,] = x[1:10,] - 2
x[11:20,] = x[11:20,] +2
t = rep(c(0,1),each=10)

# Plot the data
pos = which(t==0)
plot(x[pos,1],x[pos,2],xlim=c(-5,5),ylim=c(-5,5),xlab="x_1",ylab="x_2",pch=20)
pos = which(t==1)
points(x[pos,1],x[pos,2],xlim=c(-5,5),ylim=c(-5,5))

## Set the GP hyperparameters and compute the covariance function
alpha = 10
gamma = 0.1
N = nrow(x)
C = matrix(0,ncol=N,nrow=N)
for(n in 1:N){
  for(m in 1:N){
    C[n,m] = alpha*exp(-gamma*sum((x[n,]-x[m,])^2))
  }
}


# Newton-raphson procedure to optimise the latent function values
# Initialise all to zero
f = matrix(0,nrow=N,ncol=1)
allf = matrix(0,nrow=6,ncol=N)

# Pre-compute the inverse of C
invC = solve(C)
for(iteration in 2:6){
  g = 1/(1+exp(-f))
  gra = t - g - (invC%*%f)
  H = -diag(as.vector(g*(1-g)),nrow = nrow(g)) - invC
  f = f - solve(H)%*%gra
  allf[iteration,] = t(f)
}
H = -diag(as.vector(g*(1-g)),nrow = nrow(g)) - invC

# Plot the evolution of the f values
plot(1:6,allf[,1],xlab="Iteration",ylab="f",ylim=c(-4,4),type="l")
for(colnum in 2:N){
  points(1:6,allf[,colnum],type="l") 
}

## Plot the optimised latent function values
## Creates a 3D plot with the function value as the z axis
plotmat = rbind(x,x)
z = rbind(matrix(0,nrow=length(t),ncol=1),f)
scatterplot3d(plotmat[,1],plotmat[,2],z,xlab="x_1",ylab="x_2",
              pch=c(rep(16,20),rep(17,20)),
              color=c(rep("black",20),rep("red",20)))

## Visualise the predictive function via a large grid of test points
# Create the grid
gridseq = seq(from=-5.5,to=5.5,by=0.25)
mesh=meshgrid(gridseq)
X = mesh$X; Y = mesh$Y
testN = prod(dim(X))
testX = cbind(as.vector(X),as.vector(Y))
# Create the test covariance function
R = matrix(0,nrow=N,ncol=testN)
for(n in 1:N){
  for(m in 1:testN){
    R[n,m] = alpha*exp(-gamma*sum((x[n,] - testX[m,])^2))
  }
}

# Compute the mean predictive latent function
testf = as.vector(t(R)%*%invC%*%f)
Z = matrix(testf,ncol=ncol(X),nrow=nrow(X))


# Contour the predictions (function values, not probabilities)
filled.contour(gridseq,gridseq,t(Z),
               xlab="x_1",ylab="x_2",
               plot.axes={points(x[,1],x[,2],pch=16)}
               )

# Contour the probabilities
filled.contour(gridseq,gridseq,t(1/(1+exp(-Z))),
               xlab="x_1",ylab="x_2",
               plot.axes={points(x[,1],x[,2],pch=16)}
)

## Using the full GP distribution - propagating the uncertainity through the sigmoid
pred_var = matrix(0,nrow=testN,ncol=1)
pavg = matrix(0,nrow=length(testf),ncol=1)
minpred_var = 1e-3;
# loop over the test points, computing the marginal predictive variance and
# sampling function values before passing them through the sigmoid and
# averaging to get a probability

for(n in 1:testN){
  pred_var[n] = max(minpred_var,alpha - as.numeric(t(R[,n])%*%invC%*%R[,n]))
  u = rnorm(10000)*sqrt(pred_var[n]) + testf[n];
  pavg[n] = mean(1/(1+exp(-u)))
}

# Contour the resulting probabilities
Z = matrix(pavg,nrow=nrow(X),ncol=ncol(X))
filled.contour(gridseq,gridseq,t(Z),
               xlab="x_1",ylab="x_2",
               plot.axes={points(x[,1],x[,2],pch=16)}
)