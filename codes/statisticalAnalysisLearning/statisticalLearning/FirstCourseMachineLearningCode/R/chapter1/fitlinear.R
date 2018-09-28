## fitlinear.R
# From A First Course in Machine Learning, Chapter 1.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
rm(list=ls(all=TRUE))

## Define the data (Table 1.1)
# Change these to use a different dataset
x = c(1,3,5)
t = c(4.8,11.1,17.2)
N = length(x) # 3

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
w_0 = m_t - w_1*m_x;

## Plot the data and linear fit
plot(x,t,xlim=c(0,6),ylim=c(0,25))
xplot = c(0,6)
points(xplot,w_0+w_1*xplot,type="l")
