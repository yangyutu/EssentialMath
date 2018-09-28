## fitolympic.R
# From A First Course in Machine Learning, Chapter 1.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
rm(list=ls(all=TRUE))

## Load the Olympic data
setwd("~/data/olympics") ## Might need to change the path here
male100 = read.csv(file="male100.csv",header=FALSE)

## Extract the male 100m data
x = male100[,1] # Olympic years
t = male100[,2] # Winning times

# Change the preceeding lines for different data.  e.g.
# x = male400[,1] # Olympic years
# t = male400[,2] # Winning times
# for the mens 400m event.

N = length(x) ## 27

## Compute the various averages required
# $\frac{1}{N}\sum_n x_n$
m_x = sum(x)/N
##
# $$\frac{1}{N}\sum_n t_n$$
#
m_t = sum(t)/N
##
# $\frac{1}{N}\sum_n t_n x_n$
m_xt = sum(t*x)/N
##
# $\frac{1}{N}\sum_n x_n^2$
m_xx = sum(x*x)/N

## Compute w1 (gradient) (Equation 1.10)
w_1 = (m_xt - m_x*m_t)/(m_xx - m_x^2)
## Compute w0 (intercept) (Equation 1.8)
w_0 = m_t - w_1*m_x

## Plot the data and linear fit
plot(x,t,xlab=c("Olympic year"),ylab=c("Winning time"))
xplot = c(min(x)-4,max(x)+4)
points(xplot,w_0+w_1*xplot,type="l")




