## olymppoly.R
# From A First Course in Machine Learning, Chapter 1.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
rm(list=ls(all=TRUE))

# Load the Olympic data and extract the mens 100m data
setwd("~/data/olympics") ## Might need to change the path here
male100 = read.csv(file="male100.csv",header=FALSE)

x = male100[,1]
t = male100[,2]

# Rescale x for numerical reasons
x = x - x[1]
x = x/30

# Plot the data
plot(x,t,xlab="Olympic number (note, not year!)",ylab="Winning time")

## Linear model
plotx = seq(from=x[1]-2,to=x[length(x)]+2,by=0.01)
X = matrix(c(x^0,x^1),ncol=2,byrow=FALSE)
plotX = matrix(c(plotx^0,plotx^1),ncol=2,byrow=FALSE)

w = solve(t(X)%*%X)%*%t(X)%*%t

# Plot the model
plot(x,t,xlab="Olympic number (note, not year!)",ylab="Winning time")
points(plotx,plotX%*%w,type="l",col="blue")

## Quadratic model
plotx = seq(from=x[1]-2,to=x[length(x)]+2,by=0.01)
X = matrix(x^0,ncol=1,byrow=FALSE)
plotX = matrix(plotx^0,ncol=1,byrow=FALSE)

for(k in 1:2){
  X = cbind(X,x^k)
  plotX = cbind(plotX,plotx^k)
}

w = solve(t(X)%*%X)%*%t(X)%*%t

# Plot the model
points(plotx,plotX%*%w,type="l",col="red")

## Quartic model
plotx = seq(from=x[1]-2,to=x[length(x)]+2,by=0.01)
X = matrix(x^0,ncol=1,byrow=FALSE)
plotX = matrix(plotx^0,ncol=1,byrow=FALSE)

for(k in 1:4){
  X = cbind(X,x^k)
  plotX = cbind(plotX,plotx^k)
}
w = solve(t(X)%*%X)%*%t(X)%*%t

# Plot the model
points(plotx,plotX%*%w,type="l",col="darkgreen")

## 8th order model
plotx = seq(from=x[1]-2,to=x[length(x)]+2,by=0.01)
X = matrix(x^0,ncol=1,byrow=FALSE)
plotX = matrix(plotx^0,ncol=1,byrow=FALSE)

for(k in 1:8){
  X = cbind(X,x^k)
  plotX = cbind(plotX,plotx^k)
}
w = solve(t(X)%*%X)%*%t(X)%*%t

# Plot the model
points(plotx,plotX%*%w,type="l",col="orange")


