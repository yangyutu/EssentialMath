import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, Ridge
import seaborn as sns
sns.set()
sns.set()
mpl.rcParams.update({'font.size': 20})
mpl.rc('xtick', labelsize=20) 
mpl.rc('ytick', labelsize=20) 
plt.rcParams['xtick.labelsize']=20
plt.rcParams['ytick.labelsize']=20

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 20
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
np.random.seed(1)
plt.close('all')
np.random.seed(1)
plt.close('all')

x = np.linspace(-1, 1, 100)


X = np.c_[x.ravel(), (x**2).ravel(), (x**3).ravel(), (x**4).ravel(), (x**5).ravel(), (x**6).ravel(), (x**7).ravel(), (x**8).ravel(), (x**9).ravel(), (x**10).ravel()]

y_raw = 0.3 + 2 * x + 0.5 * x**2 - 1.6 * x**3
y = y_raw + 0.3 * np.random.randn(x.shape[0])


plt.plot(x, y, 'o', markerfacecolor='none')
plt.plot(x, y_raw)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

C = np.logspace(-3, 1, 100)

score_means = []
score_stds = []
for c in C:
    model = Ridge(alpha=c)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring = 'neg_mean_squared_error')
    score_means.append(scores.mean())
    score_stds.append(scores.std())
    
plt.figure(3, figsize=(14, 14))
plt.subplot(2,2,1)
error = -np.array(score_means)
plt.plot(np.log10(C), np.log10(error), label=r'CVError vs. $\lambda$')
plt.xlabel(r'$\log_{10}(\lambda)$')
plt.ylabel('CVError')
plt.legend()


c = 10
model = Ridge(alpha=c)
model.fit(X_train, y_train)
y_pred = model.predict(X)



plt.subplot(2,2,2)
plt.plot(x, y, 'o', markerfacecolor='none')
plt.plot(x, y_raw)
plt.plot(x, y_pred, label=r'$\lambda$=10')
plt.legend(['sample', 'true', r'model ($\lambda$=10)'])
plt.xlabel('x')
plt.ylabel('y')


c = 0.1
model = Ridge(alpha=c)
model.fit(X_train, y_train)
y_pred = model.predict(X)

plt.subplot(2,2,3)
plt.plot(x, y, 'o', markerfacecolor='none')
plt.plot(x, y_raw)
plt.plot(x, y_pred, label=r'$\lambda = 0.1$')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

c = 0.001
model = Ridge(alpha=c)
model.fit(X_train, y_train)
y_pred = model.predict(X)

plt.subplot(2,2,4)
plt.plot(x, y, 'o', markerfacecolor='none')
plt.plot(x, y_raw)
plt.plot(x, y_pred, label=r'$\lambda = 0.001$')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()