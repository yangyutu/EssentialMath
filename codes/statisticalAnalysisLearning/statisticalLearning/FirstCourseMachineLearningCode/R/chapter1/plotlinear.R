## plotlinear.R
# From A First Course in Machine Learning, Chapter 1.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
rm(list=ls(all=TRUE))

## Define two points for the x-axis
x = c(-5,5)

## Define the different intercepts and gradients to plot
w0 = seq(from=0,to=20,by=1)
w1 = seq(from=0,to=8,by=0.4)

## Plot all of the lines
plot(x,w0[1]+w1[1]*x,type="l",xlim=c(-5,5),ylim=c(-20,60),ylab="y = w0 + w1*x")
cat("y = 0 + 0 x")
for(i in 1:length(w0)){
  points(x,w0[i]+w1[i]*x,type="l")
  cat(paste("y =",w0[i],"+",w1[i],"x \n"))
}

## Request user input
cat("\n Keeps plotting lines on the current plot until you quit Esc\n")
plot(x,0+0*x,type="l",xlim=c(-5,5),ylim=c(-20,60),ylab="y = w0 + w1*x")
while(TRUE){ 
  intercept <- as.numeric(readline(prompt="Enter intercept: "))
  gradient <- as.numeric(readline(prompt="Enter gradient: "))
  points(x,intercept + gradient*x,type="l",xlim=c(-5,5),ylim=c(-20,60))
  cat(paste("y =",intercept,"+",gradient,"x \n"))
}
