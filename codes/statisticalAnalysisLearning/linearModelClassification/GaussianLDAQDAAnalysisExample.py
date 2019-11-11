from scipy import linalg
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from matplotlib import cm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import PolynomialFeatures
from mpl_toolkits.mplot3d import Axes3D 

plt.rcParams.update({'font.size': 18})
plt.rc('font', family='serif')
plt.close('all')
# #############################################################################
# Colormap
cmap = colors.LinearSegmentedColormap(
    'red_blue_classes',
    {'red': [(0, 1, 1), (1, 0.7, 0.7)],
     'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
     'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
plt.cm.register_cmap(cmap=cmap)
from matplotlib.colors import ListedColormap
cm_bright = ListedColormap(['#bb4444', '#4444bb'])


'''Generate 2 Gaussians samples with different covariance matrices'''
n, dim = 300, 2
np.random.seed(0)
C = np.array([[0., -1.], [2.5, .3]]) * 2.
X = np.r_[np.dot(np.random.randn(n, dim), C),
          np.dot(np.random.randn(n, dim), C.T) + np.array([1, 4])]
y = np.hstack((np.zeros(n), np.ones(n)))


X0, X1 = X[y == 0], X[y == 1]

fig = plt.figure(1, figsize=(10, 10))
ax = plt.subplot(2,2,1)

ax.scatter(X0[:, 0], X0[:, 1], marker='.', color='red')
ax.scatter(X1[:, 0], X1[:, 1], marker='.', color='blue')
plt.title('raw data')


lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
lda.fit(X, y)

# Quadratic Discriminant Analysis
qda = QuadraticDiscriminantAnalysis(store_covariance=True)
qda.fit(X, y)

# class 0 and 1 : areas
nx, ny = 200, 100
x_min, x_max = plt.xlim()
y_min, y_max = plt.ylim()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                     np.linspace(y_min, y_max, ny))

Z_LDA = lda.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z_LDA.shape = xx.shape
ax = plt.subplot(2,2,2)

ax.contourf(xx, yy, Z_LDA, cmap = cm_bright)
ax.scatter(X0[:, 0], X0[:, 1], marker='.', color='red')
ax.scatter(X1[:, 0], X1[:, 1], marker='.', color='blue')
plt.title('LDA')


Z_QDA = qda.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z_QDA.shape = xx.shape
ax = plt.subplot(2,2,3)

ax.contourf(xx, yy, Z_QDA, cmap =cm_bright)
ax.scatter(X0[:, 0], X0[:, 1], marker='.', color='red')
ax.scatter(X1[:, 0], X1[:, 1], marker='.', color='blue')
plt.title('QDA')


poly = PolynomialFeatures(include_bias = False)

X_poly = poly.fit_transform(X)
lda_poly = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
lda_poly.fit(X_poly, y)

Z_LDA_poly = qda.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z_LDA_poly.shape = xx.shape
ax = plt.subplot(2,2,4)

ax.contourf(xx, yy, Z_LDA_poly, cmap =cm_bright)
ax.scatter(X0[:, 0], X0[:, 1], marker='.', color='red')
ax.scatter(X1[:, 0], X1[:, 1], marker='.', color='blue')
plt.title('LDA-poly')



fig = plt.figure(2, figsize=(12, 12))
ax = plt.subplot(2, 2, 1, projection='3d')
surf = ax.plot_surface(xx, yy, Z_LDA, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

plt.title('LDA')
ax = plt.subplot(2, 2, 2, projection='3d')
surf = ax.plot_surface(xx, yy, Z_QDA, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

plt.title('QDA')

ax = plt.subplot(2, 2, 3, projection='3d')
surf = ax.plot_surface(xx, yy, Z_LDA_poly, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

plt.title('LDA-poly')

X_vec = np.c_[xx.ravel(), yy.ravel()]

delta0 = np.diagonal(-0.5 * (X_vec - qda.means_[0]) @ np.linalg.inv(qda.covariance_[0]) @ (X_vec - qda.means_[0]).T)
delta0.shape = xx.shape
ax = plt.subplot(2, 2, 4, projection='3d')
surf = ax.plot_surface(xx, yy, delta0, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

delta1 = np.diagonal(-0.5 * (X_vec - qda.means_[1]) @ np.linalg.inv(qda.covariance_[1]) @ (X_vec - qda.means_[1]).T)
delta1.shape = xx.shape
surf = ax.plot_surface(xx, yy, delta1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.title('QDA')
