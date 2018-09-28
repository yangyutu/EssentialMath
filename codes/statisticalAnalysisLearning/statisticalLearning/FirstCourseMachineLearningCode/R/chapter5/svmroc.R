## svmroc.R
# From A First Course in Machine Learning, Chapter 5.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
# ROC analysis of SVM
rm(list=ls(all=TRUE))
require(quadprog) # might require to install this package: install.packages("quadprog")
require(pracma) # might require to install this package: install.packages("pracma")

## Load the data
setwd("~/data/SVMdata2") ## Might need to change the path here
X = as.matrix(read.csv(file="X.csv",header=FALSE))
t = as.matrix(read.csv(file="t.csv",header=FALSE))

setwd("~/data/SVMtest") ## Might need to change the path here
testX = as.matrix(read.csv(file="testX.csv",header=FALSE))
testt = as.matrix(read.csv(file="testt.csv",header=FALSE))

## Compute Kernel and test Kernel
N =nrow(X)
Nt = nrow(testX)
K = matrix(0,nrow=N,ncol=N)
testK = matrix(0,nrow=N,ncol=Nt)

gam=10 # Experiment with this value
for(n in 1:N){
  for(n2 in 1:N){
    K[n,n2] = exp(-gam*sum((X[n,]-X[n2,])^2))
  }
  for(n2 in 1:Nt){
    testK[n,n2] = exp(-gam*sum((X[n,]-testX[n2,])^2))
  }
}

## Setup the SVM optimisation problem in the matrix form below
# $$ \min \frac{1}{2}b^T D b - d^T b$$
# s.t. $$ A^T b \geq b_0 $
# where the first inequality is treated as an equality.
D = (t%*%t(t))*K + (1e-5)*diag(N)
d = matrix(rep(1,N))

# Fix C
C = 10
A = cbind(t,diag(N),(-1)*diag(N))
b0 = c(rep(0,N+1),rep(-C,N))

# Following line runs the SVM optimisation
alpha = solve.QP(Dmat=D,dvec=d,Amat=A,bvec=b0,meq=1)

# Compute the bias
fout = colSums(matrix(alpha$solution*t,ncol=N,nrow=N,byrow=FALSE)*K)
pos = which(alpha$solution>1e-3)
bias = mean(t[pos]-fout[pos])

# Compute the test predictions
testpred = t(alpha$solution*t)%*%testK + bias
testpred = t(testpred)

## Do the ROC analysis
th_vals = seq(min(testpred),max(testpred)+0.01,0.01)
sens = rep(NA,length(th_vals))
spec = rep(NA,length(th_vals))
for(i in 1:length(th_vals)){
  b_pred = as.vector(testpred)>=th_vals[i]
  # Compute true positives, false positives, true negatives, true
  # positives
  TP = sum(b_pred==1 & testt == 1)
  FP = sum(b_pred==1 & testt == -1)
  TN = sum(b_pred==0 & testt == -1)
  FN = sum(b_pred==0 & testt == 1)
  # Compute sensitivity and specificity
  sens[i] = TP/(TP+FN)
  spec[i] = TN/(TN+FP)
}

## Plot the ROC curve
cspec = 1-spec
cspec = cspec[length(cspec):1]
sens = sens[length(cspec):1]
plot(cspec,sens,type="l",main="ROC")

## Compute the AUC
AUC = sum(0.5*(sens[2:length(sens)]+sens[1:(length(sens)-1)])*(cspec[2:length(cspec)] - cspec[1:(length(cspec)-1)]))
cat('\nAUC:',AUC)
