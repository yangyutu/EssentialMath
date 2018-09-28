## synthquad.R
# From A First Course in Machine Learning, Chapter 1.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
rm(list=ls(all=TRUE))

## Generate a synthetic dataset from a quadratic function
N = 200 # Number of data points
# Generate random x values between -5 and 5
x = runif(n=N,min=-5,max=5)

## Define the function and the true parameters
# $t = w_0 + w_1x + w_2x^2$
w_0 = 1
w_1 = -2
w_2 = 0.5

## Define t
t = w_0 + w_1*x + w_2*(x^2)

## Add some noise
t = t + 0.5*rnorm(n=N)

## Plot the data
plot(x,t)

## Fit the quadratic model and a linear model for comparison
X = matrix(c(x^0,x^1,x^2),ncol=3,byrow=FALSE)
for(k in c(1,2)){
  if(k==1){
    linear_w = (solve(t(X[,c(1,2)])%*%X[,c(1,2)])%*%t(X[,c(1,2)]))%*%t
  }
  if(k==2){
    quad_w = (solve(t(X)%*%X)%*%t(X))%*%t
  }
}
cat(paste("\n Linear function: t =",linear_w[1,],"+",linear_w[2,],"x"))
cat(paste("\n Quadratic function: t =",quad_w[1,],"+",quad_w[2,],"x +",quad_w[3,],"x^2"))

## Plot the functions
plotx = seq(from=-5,to=5,by=0.01)
plotX = matrix(c(plotx^0,plotx^1,plotx^2),ncol=3,byrow=FALSE)
plot(x,t)
points(plotx,plotX%*%quad_w,type="l",col="blue")
points(plotx,plotX[,c(1,2)]%*%linear_w,type="l",col="red")
legend(2.5,23,c("Quadratic","Linear"),lty=c(1,1),lwd=c(2.5,2.5),col=c("blue","red"))
