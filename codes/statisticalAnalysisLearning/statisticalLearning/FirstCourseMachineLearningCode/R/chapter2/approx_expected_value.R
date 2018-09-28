## approx_expected_value.R
# From A First Course in Machine Learning, Chapter 2.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
# Approximating expected values via sampling
rm(list=ls(all=TRUE))

## We are trying to compute the expected value of
# $y^2$
#
# Where
# $p(y)=U(0,1)$
#
# Which is given by:
# $\int y^2 p(y) dy$
##
# The analytic result is:
# $\frac{1}{3}$
## Generate samples
ys = runif(10000)
# compute the expectation
ey2 = mean(ys^2)
cat(paste("\nSample-based approximation:",ey2))
## Look at the evolution of the approximation
posns = seq(1,length(ys),10)
ey2_evol = rep(0,length(posns))
for(i in 1:length(posns)){
  ey2_evol[i] = mean(ys[1:posns[i]]^2)
}
plot(posns,ey2_evol,xlab='Samples',ylab='Approximation',ylim=c(0.28,0.42))
lines(c(posns[1]-1000,posns[length(posns)]+1000),c(1/3,1/3),lty=3,col="red")



