## olympval.R
# From A First Course in Machine Learning, Chapter 1.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
rm(list=ls(all=TRUE))

# Load the Olympic data and extract the training and validation data
setwd("~/data/olympics") ## Might need to change the path here
male100 = read.csv(file="male100.csv",header=FALSE)

x = male100[,1]
t = male100[,2]
pos = which(x>1979)

# Rescale x for numerical reasons
x = x - x[1]
x = x/30

valx = x[pos]
valt = t[pos]
x = x[-pos] 
t = t[-pos] 

## Fit the different models and plot the results
orders = c(1,4,8) #We shall fit models of these orders

# Plot the data
plot(x,t,xlim=c(min(x,valx)-0.2,max(x,valx)+0.2),ylim=c(9,12))
points(valx,valt,col="blue")
plotx = seq(min(x),max(valx),0.01)
val_loss <- rep(NA,length(orders))
for(i in 1:length(orders)){
    X = matrix(x^0,ncol=1)
    plotX = matrix(plotx^0,ncol=1)
    valX = matrix(valx^0,ncol=1)
    for(k in 1:orders[i]){
        X = cbind(X,x^k)
        valX = cbind(valX,valx^k)
        plotX = cbind(plotX,plotx^k)
    }
    # Compute w
    w = solve(t(X)%*%X)%*%t(X)%*%t
    points(plotx,plotX%*%w,type="l",col=i)
    
    # Compute validation loss
    val_loss[i] = mean((valX%*%w - valt)^2)
}
legend(x=0,y=10,legend=c('Training','Validation','Linear','4th order','8th order'),
       pch=c(1,1,NA,NA,NA),lty=c(NA,NA,1,1,1),col=c("black","blue","black","green","red"),merge=TRUE)
for(i in 1:length(orders)){
  cat(paste("\n Model order: ",orders[i],", Validation loss: ",val_loss[i],sep=""))
}
