import numpy as np

class SimpleSVM(object):
	def __init__(self,trainx,traint,kernel='rbf',kpar = 1.0,C = 2,tol=1e-10,max_passes = 1000):
		self.trainx = trainx
		self.traint = traint
		self.kernel = kernel
		self.kpar = kpar
		self.C = C
		self.tol = tol
		self.max_passes = max_passes
		self.N = len(self.traint)
		self.K = self.kernel_matrix()
		self.alpha = np.zeros_like(traint)
		self.b = 0.0

	def test_kernel(self,testx):
		testK = np.zeros_like(self.traint)
		if self.kernel == 'linear':
			testK = np.dot(self.trainx,testx.T)[:,None]
		elif self.kernel == 'rbf':
			for i in range(self.N):
				testK[i] = np.exp(-self.kpar * ((self.trainx[i,:]-testx)**2).sum())
		return testK

	def kernel_matrix(self):
		K = None
		if self.kernel == 'linear':
			K = np.dot(self.trainx,self.trainx.T)
		elif self.kernel == 'rbf':
			K = np.zeros((self.N,self.N))
			for i in range(self.N):
				for j in range(self.N):
					K[i,j] = np.exp(-self.kpar * ((self.trainx[i,:]-self.trainx[j,:])**2).sum())
		return K

	def train_predict(self,i):
		ksub = self.K[i,:][:,None]
		return (ksub*self.alpha*self.traint).sum() + self.b


	def test_predict(self,testx):
		testK = self.test_kernel(testx)	
		return (testK*self.alpha*self.traint).sum() + self.b

	def smo_optimise(self):
	    # initialise
	    self.alpha = np.zeros_like(self.traint)
	    self.b = 0
	    passes = 0
	    while passes < self.max_passes:
	        num_changed_alphas = 0
	        for i in range(self.N):
	            Ei = self.train_predict(i) - self.traint[i]
	            if (self.traint[i]*Ei < -self.tol and self.alpha[i] < self.C) or (self.traint[i]*Ei > self.tol and self.alpha[i] > 0):
	                j = np.random.randint(self.N)
	                Ej = self.train_predict(j) - self.traint[j]
	                alphai = float(self.alpha[i])
	                alphaj = float(self.alpha[j])
	                if self.traint[i] == self.traint[j]:
	                    L = max((0,alphai+alphaj-self.C))
	                    H = min((self.C,alphai+alphaj))
	                else:
	                    L = max((0,alphaj-alphai))
	                    H = min((self.C,self.C+alphaj-alphai))
	                if L==H:
	                    continue

	                eta = 2*self.K[i,j] - self.K[i,i] - self.K[j,j]
	                if eta >= 0:
	                    continue


	                self.alpha[j] = alphaj - (self.traint[j]*(Ei-Ej))/eta
	                if self.alpha[j] > H:
	                    self.alpha[j] = H
	                if self.alpha[j] < L:
	                    self.alpha[j] = L

	                if abs(self.alpha[j]-alphaj) < 1e-5:
	                    continue

	                self.alpha[i] = self.alpha[i] + self.traint[i]*self.traint[j]*(alphaj - self.alpha[j])
	                
	                b1 = self.b - Ei - self.traint[i]*(self.alpha[i] - alphai)*self.K[i,i] - self.traint[j]*(self.alpha[j] - alphaj)*self.K[i,j]
	                b2 = self.b - Ej - self.traint[i]*(self.alpha[i] - alphai)*self.K[i,j] - self.traint[j]*(self.alpha[j] - alphaj)*self.K[j,j]
	                if self.alpha[i]>0 and self.alpha[i]<self.C:
	                    self.b = b1
	                elif self.alpha[j]>0 and self.alpha[j]<self.C:
	                    self.b = b2
	                else:
	                    self.b = (b1+b2)/2.0
	                num_changed_alphas += 1
	            if num_changed_alphas == 0:
	                passes += 1
	            else:
	                passes = 0
