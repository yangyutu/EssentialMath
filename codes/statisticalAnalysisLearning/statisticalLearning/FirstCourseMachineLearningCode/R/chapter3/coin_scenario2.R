## coin_scenario2.R
# From A First Course in Machine Learning, Chapter 2.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
# Coin game, prior scenario 2
rm(list=ls(all=TRUE))

## Load the coin data
setwd("~/data/coin_data") ## Might need to change the path here
big_data = as.vector(unlist(read.csv(file="big_data.csv",header=FALSE)))
toss_data = as.vector(unlist(read.csv(file="toss_data.csv",header=FALSE)))
toss_data2 = as.vector(unlist(read.csv(file="toss_data2.csv",header=FALSE)))

## Plot the prior density
alpha = 50
beta = 50
cat('\nPrior parameters: alpha:',alpha,'beta:',beta)
r = seq(0,1,0.01)
plot(r,dbeta(r,alpha,beta),type="l",xlab="r",ylab="p(r)",ylim=c(0,10))

## Incorporate the data one toss at a time
post_alpha = alpha
post_beta = beta
ch = c('T','H')
toss_string = c()
for(i in 1:length(toss_data)){
  ##
  toss_string = c(toss_string,ch[toss_data[i]+1])
  plot(r,dbeta(r,post_alpha,post_beta),type="l",ylim=c(0,10),lty=2,
       xlab="r",ylab="p(r|...)",main=paste("Posterior after",i,"tosses"))
  post_alpha = post_alpha + toss_data[i]
  post_beta = post_beta + 1 - toss_data[i]
  points(r,dbeta(r,post_alpha,post_beta),type="l");
}

## Incorporate another ten
plot(r,dbeta(r,post_alpha,post_beta),type="l",ylim=c(0,10),lty=2,
     xlab="r",ylab="p(r|...)",main="Posterior after 20 tosses")
N = length(toss_data2)
post_alpha = post_alpha + sum(toss_data2)
post_beta = post_beta + N - sum(toss_data2)
points(r,dbeta(r,post_alpha,post_beta),type="l",ylim=c(0,10),
       xlab="r",ylab="p(r|...)")

## Incorpoate another 1000
plot(r,dbeta(r,post_alpha,post_beta),type="l",ylim=c(0,10),lty=2,
     xlab="r",ylab="p(r|...)",main="Posterior after 1020 tosses")
N = length(big_data)
post_alpha = post_alpha + sum(big_data)
post_beta = post_beta + N - sum(big_data)
points(r,dbeta(r,post_alpha,post_beta),type="l",ylim=c(0,10),
       xlab="r",ylab="p(r|...)")

## Interactive example
cat('\n Enter H or T to add a toss result and see the effect on the posterior.  Use ctrl-C to exit');
nTosses = 0
post_alpha = alpha
post_beta = beta
while(TRUE){
  cat(paste('\nYou have currently entered %g tosses\n',nTosses))
  this_toss = readline(prompt='Enter next toss ("H" or "T"):')
  plot(r,dbeta(r,post_alpha,post_beta),type="l",ylim=c(0,10),lty=2,
       xlab="r",ylab="p(r|...)")
  if(this_toss=="H"){
    post_alpha = post_alpha + 1
  }
  if(this_toss=="T"){
    post_beta = post_beta + 1;
  }
  nTosses = nTosses + 1
  points(r,dbeta(r,post_alpha,post_beta),type="l")
}
