## knnexample.R
# From A First Course in Machine Learning, Chapter 4.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
# Setting K in KNN
rm(list=ls(all=TRUE))

## Generate some data
N1 = 100
N2 = 30 # Class sizes
x = rbind(matrix(rnorm(N1*2),ncol=2),matrix(rnorm(N2*2)+3,ncol=2))
t = matrix(c(rep(0,N1),rep(1,N2)),ncol=1)
N = nrow(x)

## Plot the data
tv = as.vector(unique(t))
ma = c(0,16)
col= c("blue","red")
pos = which(t==tv[1])
plot(x[pos,1],x[pos,2],col=col[1],pch=ma[1],xlim=c(-3,6),ylim=c(-3,6))
for(i in 1:length(tv)){
  pos = which(t==tv[i])
  points(x[pos,1],x[pos,2],col=col[i],pch=ma[i])
}

## Generate the decision boundaries for various values of K
gridvals = seq(from=min(x),to=max(x),by=0.2)
mesh = meshgrid(gridvals,gridvals)
Xv = mesh$X; Yv=mesh$Y

# Loop over test points
Kvals = c(1,2,5,10,20,50,59)
for(kv in 1:length(Kvals)){
  ##
  classes = matrix(0,ncol=dim(Xv)[1],nrow=dim(Xv)[2])
  K = Kvals[kv]
  for(i in 1:nrow(Xv)){
    for(j in 1:ncol(Xv)){
      this = c(Xv[i,j],Yv[i,j])
      dists = rowSums((x - matrix(rep(this,N),ncol=2))^2)
      d = sort(dists)
      I = order(dists)
      tab = table(t[I[1:K],])
      a = as.vector(tab)
      b = as.numeric(names(tab)) 
      pos = which(a==max(a))
      if(length(pos)>1){
        ordering = sample.int(length(pos))
        pos = pos[ordering[1]]
      }
      classes[i,j] = b[pos]
    }
  }
  pos = which(t==tv[1])
  maintitle = paste("K=",K)
  plot(x[pos,1],x[pos,2],col=col[1],pch=ma[1],xlim=c(min(x[,1]),max(x[,2])),
       ylim=c(min(x[,2]),max(x[,2])),main=maintitle)
  for(i in 1:length(tv)){
    pos = which(t==tv[i])
    points(x[pos,1],x[pos,2],col=col[i],pch=ma[i])
  }
  contour(gridvals,gridvals,classes,add=TRUE)
}








