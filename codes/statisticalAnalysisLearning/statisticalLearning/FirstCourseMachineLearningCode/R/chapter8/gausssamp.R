gausssamp <- function(mu,sigma,N,sigmachol=FALSE){
  if(is.logical(sigmachol)){
    sigmachol = chol(sigma)
    sigmachol=t(sigmachol)
  }
  q = matrix(rnorm(nrow(mu)*N),nrow=nrow(mu),ncol=N)
  g = matrix(rep(mu,N),ncol=N)+(sigmachol%*%q)
  g = t(g)
  return(g)
}