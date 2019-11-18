import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
import seaborn as sns
plt.close('all')
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



xmin, xmax = -3, 3
xx = np.linspace(xmin, xmax, 100)
lw = 2
plt.figure(figsize=(8, 8))
plt.plot([xmin, 0, 0, xmax], [1, 1, 0, 0], color='red', lw=lw,
         label="Zero-one loss")
plt.plot(xx, np.where(xx < 1, 1 - xx, 0), color='black', lw=lw,
         label="Hinge loss")
plt.plot(xx, -np.minimum(xx, 0), color='yellowgreen', lw=lw,
         label="Perceptron loss")
plt.plot(xx, np.log(1 + np.exp(-xx)), color='cornflowerblue', lw=lw,
         label="CrossEntroy loss")
plt.plot(xx, np.square(1 - xx), color='magenta', lw=lw,
         label="Squared loss")
plt.ylim((0, 4))
plt.legend(loc="upper right")
plt.xlabel(r"$yf(x)$")
plt.ylabel("$Loss$")
plt.show()