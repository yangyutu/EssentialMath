## knncv.R
# From A First Course in Machine Learning, Chapter 5.
# Francois-Xavier Briol, 04/06/16 [f-x.briol@warwick.ac.uk]
# Cross-validation over K in KNN
rm(list=ls(all=TRUE))

## Generate some data
N1 = 100
N2 = 20 # Class sizes
x = rbind(matrix(rnorm(N1*2),ncol=2),matrix(rnorm(N2*2)+2,ncol=2))
t = matrix(c(rep(0,N1),rep(1,N2)),ncol=1)
N = nrow(x)

## Plot the data
tv = as.vector(unique(t))
ma = c(0,16)
col= c("blue","red")
pos = which(t==tv[1])
plot(x[pos,1],x[pos,2],col=col[1],pch=ma[1],xlim=c(-3,5),ylim=c(-3,5))
for(i in 1:length(tv)){
  pos = which(t==tv[i])
  points(x[pos,1],x[pos,2],col=col[i],pch=ma[i])
}

## loop over values of K
Nfold = 10
Kvals = seq(1,30,2)
Nrep = 100
Errors = list()
for(rep in 1:Nrep){
  Errors[[rep]] <- matrix(0,nrow=length(Kvals),ncol=Nfold)
  ## Permute the data and split into folds
  ordering = sample.int(N)
  Nfold = 10 # 10-fold CV
  sizes = rep(floor(N/Nfold),Nfold)
  sizes[length(sizes)] = sizes[length(sizes)] + N - sum(sizes)
  csizes = c(0,cumsum(sizes))
  for(kv in 1:length(Kvals)){
    K = Kvals[kv]
    # Loop over folds
    for(fold in 1:Nfold){
      trainX = x
      traint = t
      foldindex = ordering[(csizes[fold]+1):csizes[fold+1]]
      trainX = trainX[-foldindex,]
      traint = traint[-foldindex]
      testX = x[foldindex,]
      testt = t[foldindex]
      
      # Do the KNN
      classes = rep(0,nrow(testX))
      for(i in 1:size(testX,1)){
        this = testX[i,]
        dists = rowSums((trainX - matrix(rep(this,nrow(trainX)),ncol=2,byrow=TRUE))^2)
        d = sort(dists)
        I = order(dists)
        tab = table(traint[I[1:K]])
        a = as.vector(tab)
        b = as.numeric(names(tab)) 
        pos = which(a==max(a))
        if(length(pos)>1){
          temp_order = sample.int(length(pos))
          pos = pos[temp_order[1]]
        }
        classes[i] = b[pos]
      }
    Errors[[rep]][kv,fold] = sum(as.vector(classes)!=testt)
    }
  }
}

## Plot the results
sumErrors = matrix(0,nrow=length(Kvals),ncol=Nfold)
for(rep in 1:Nrep){
  sumErrors = sumErrors + Errors[[rep]]
}
s = rowSums(sumErrors)/(N*Nrep)
plot(Kvals,s,xlab="K",ylab="Error",type="l")

