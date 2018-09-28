## olymppred.R
# From A First Course in Machine Learning, Chapter 1.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
rm(list=ls(all=TRUE))

# Load Olympic data and fit linear model (see fitolympic.m)
setwd("~/data/olympics") ## Might need to change the path here
male100 = read.csv(file="male100.csv",header=FALSE)
x = male100[,1] # Olympic years
t = male100[,2] # Winning times
N = length(x) # 28
m_x = sum(x)/N
m_t = sum(t)/N
m_xt = sum(t*x)/N
m_xx = sum(x*x)/N
w_1 = (m_xt - m_x*m_t)/(m_xx - m_x^2)
w_0 = m_t - w_1*m_x
plot(x,t,xlab=c("Olympic year"),ylab=c("Winning time"),xlim=c(1890,2020),ylim=c(9,12))
xplot = c(min(x)-10,max(x)+10)
points(xplot,w_0+w_1*xplot,type="l")

## Make predictions at 2012 and 2016
xpred = c(2016,2020) # Add more values to this vector for more predictions
tpred = w_0 + w_1*xpred

## Display the predictions
for(i in 1:length(xpred)){
  cat(paste("\n Predicted winning time in",xpred[i],"is:",tpred[i],"seconds"))
}

## Add predictions to the plot
points(xpred,tpred,col="red")
