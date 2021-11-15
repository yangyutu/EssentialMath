import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Lasso, Ridge, LinearRegression
import seaborn as sns
from sklearn.metrics import mean_squared_error
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


X = np.c_[np.ones_like(x.ravel()), x.ravel(), (x**2).ravel(), (x**3).ravel(), (x**4).ravel(),
          (x**5).ravel(), (x**6).ravel(), (x**7).ravel(),  (x**8).ravel(), (x**9).ravel(),
          (x**10).ravel(), (x**11).ravel()]

y_raw = 0.3 + 2 * x + 0.8 * x**2 - 1.6 * x**3
np.random.seed(2)
y = y_raw + 0.2 * np.random.randn(x.shape[0])
y_test = y_raw + 0.2 * np.random.randn(x.shape[0])

plt.figure(1, figsize=(17, 5))
plt.subplot(1,3,1)
plt.plot(x, y, 'o', markerfacecolor='none', markeredgecolor='r', label='train')
plt.plot(x, y_test, '^', markerfacecolor='none', markeredgecolor='g', label='test')
plt.plot(x, y_raw, linewidth=2, color='black')
err = mean_squared_error(y_raw, y)
test_err = mean_squared_error(y_raw, y_test)
print(err, test_err)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Ground truth\n train mse={err:0.3f}, test mse={test_err:0.3f}')


model = LinearRegression()
model.fit(X[:,0:2], y)
y_pred = model.predict(X[:,0:2])
train_err = mean_squared_error(y_pred, y)
test_err = mean_squared_error(y_pred, y_test)
print(train_err, test_err)
plt.subplot(1,3,2)
plt.plot(x, y, 'o', markerfacecolor='none', markeredgecolor='r')
plt.plot(x, y_test, '^', markerfacecolor='none', markeredgecolor='g')
plt.plot(x, y_pred, linewidth=2, color='black')
plt.xlabel('x')
plt.title(f'Polynomial d=1 \n train mse={train_err:0.3f}, test mse={test_err:0.3f}')


# model.fit(X[:,0:5], y)
# y_pred = model.predict(X[:,0:5])
# err = np.linalg.norm(y_pred - y)
# print(err)
# plt.subplot(2,2,3)
# plt.plot(x, y, 'o', markerfacecolor='none')
# plt.plot(x, y_pred)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title(f'polynomial d=3, mse={err:0.2f}')



model.fit(X[:,0:11], y)
y_pred = model.predict(X[:,0:11])
train_err = mean_squared_error(y_pred, y)
test_err = mean_squared_error(y_pred, y_test)
print(train_err, test_err)
plt.subplot(1,3,3)
plt.plot(x, y, 'o',  markerfacecolor='none',markeredgecolor='r')
plt.plot(x, y_test, '^', markerfacecolor='none', markeredgecolor='g')
plt.plot(x, y_pred, linewidth=2, color='black')
plt.xlabel('x')
plt.title(f'Polynomial d=9 \n train mse={train_err:0.3f}, test mse={test_err:0.3f}')
plt.savefig('overfitting_demo.png', dpi=300)