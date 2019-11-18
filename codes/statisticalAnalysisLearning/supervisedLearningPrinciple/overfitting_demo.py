import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, Ridge, LinearRegression
import seaborn as sns
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

x = np.linspace(-1, 1, 20)


X = np.c_[x.ravel(), (x**2).ravel(), (x**3).ravel(), (x**4).ravel(), (x**5).ravel(), (x**6).ravel(), (x**7).ravel(),  (x**8).ravel(), (x**9).ravel()]

y_raw = 0.3 + 2 * x + 0.8 * x**2 - 1.6 * x**3
y = y_raw + 0.2 * np.random.randn(x.shape[0])


plt.figure(1, figsize=(10, 10))
plt.subplot(2,2,1)
plt.plot(x, y, 'o', markerfacecolor='none')
plt.plot(x, y_raw)
plt.ylabel('y')
plt.title('ground truth')


model = LinearRegression()
model.fit(X[:,0].reshape(-1,1), y)
y_pred = model.predict(X[:,0].reshape(-1,1))

plt.subplot(2,2,2)
plt.plot(x, y, 'o', markerfacecolor='none')
plt.plot(x, y_pred)
plt.ylabel('y')
plt.title('polynomial d=1')


model.fit(X[:,0:3], y)
y_pred = model.predict(X[:,0:3])

plt.subplot(2,2,3)
plt.plot(x, y, 'o', markerfacecolor='none')
plt.plot(x, y_pred)
plt.xlabel('x')
plt.ylabel('y')
plt.title('polynomial d=3')



model.fit(X[:,0:9], y)
y_pred = model.predict(X[:,0:9])
plt.subplot(2,2,4)
plt.plot(x, y, 'o', markerfacecolor='none')
plt.plot(x, y_pred)
plt.xlabel('x')
plt.ylabel('y')
plt.title('polynomial d=7')