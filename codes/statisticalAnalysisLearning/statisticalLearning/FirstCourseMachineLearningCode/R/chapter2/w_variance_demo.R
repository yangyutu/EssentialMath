## w_variation_demo.R
# From A First Course in Machine Learning, Chapter 2.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
# The bias in the estimate of the variance
# Generate lots of datasets and look at how the average fitted variance
# agrees with the theoretical value
rm(list=ls(all=TRUE))

## Generate the datasets and fit the parameters
true_w = matrix(c(-2,3),ncol=1)
Nsizes = seq(20,1000,20)
N_data = 1000 # Number of datasets
all_ss = matrix(ncol=N_data,nrow=length(Nsizes))
for(j in 1:length(Nsizes)){
  N = Nsizes[j] # Number of objects
  x = runif(N)
  X = cbind(x^0,x^1)
  noisevar = 0.5^2
  for(i in 1:N_data){
    t = X%*%true_w + rnorm(N)*sqrt(noisevar)
    w = solve(t(X)%*%X)%*%t(X)%*%t
    ss = as.numeric((1/N)*(t(t)%*%t - t(t)%*%X%*%w))
    all_ss[j,i] = ss
  }
}

## The expected value of the fitted variance is equal to:
# $\sigma^2\left(1-\frac{D}{N}\right)$
# where $D$ is the number of dimensions (2) and $\sigma^2$ is the true variance.
# Plot the average empirical value of the variance against the 
# theoretical expected value as the size of the datasets increases

plot(Nsizes,apply(X=all_ss,MARGIN=1,FUN=mean),xlab="Dataset size",ylab="Variance")
points(Nsizes,noisevar*(1-2./Nsizes),col="red",type="l")
legend(x=600,y=0.235,legend=c('Empirical','Theoretical'),col=c("black","red"),pch=c(1,NA),lty=c(NA,1),merge=TRUE)
