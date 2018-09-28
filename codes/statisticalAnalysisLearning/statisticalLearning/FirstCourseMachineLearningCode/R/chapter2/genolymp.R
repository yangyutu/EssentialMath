## genolymp.R
# From A First Course in Machine Learning, Chapter 2.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
# Generative model for the Olympic data
rm(list=ls(all=TRUE))

## Define the parameters - from previous exercises (see fitolympic.R)
w = c(36.30912,-0.01327)

## Load the Olympic data to get the Olympic years
setwd("~/data/olympics") ## Might need to change the path here
male100 = read.csv(file="male100.csv",header=FALSE)
x = male100[,1] # Olympic years

## Compute the means from the model
X = cbind(x^0,x^1)
mean_t = X%*%w

## Plot the means
plot(x,mean_t,xlab=c('Olympic year'),ylab=c('Winning time'),ylim=c(9,12))

## Generate some noise
noise_var = 0.01 # Vary this to change the noise level
noisy_t = mean_t + rnorm(length(mean_t))*sqrt(noise_var)

## Add these to the plot
points(x,noisy_t)
# Draw lines between means and noise
for(i in 1:length(x)){
  lines(c(x[i],x[i]),c(mean_t[i],noisy_t[i]));
}

## Add the actual data as red dots, for comparison
points(x,male100[,2],col="red")

