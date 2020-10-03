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
plt.figure(1, figsize = (14,14))
plt.subplot(221)
t = 2
plt.plot(xx, 0.5*xx**2, color='red', lw=lw, label="LS")
plt.plot(xx, np.where(np.abs(xx) < t , 0.5*((xx)**2), t*np.abs(xx) - 0.5*(t**2)), color='blue', lw=lw, label="Huber")
plt.plot(xx, np.where(np.abs(xx) < t , np.square(xx)/6.0 * (1 - (1 - np.square((xx)/t))**3), np.square(t)/6.0), color='green', lw=lw, label="Bisquare")

ax1 = plt.gca()
ax1.set_xlabel('Error e')
ax1.set_ylabel(r'$\rho(e)$')

plt.subplot(222)
plt.plot(xx, np.ones_like(xx) * xx, color='red', lw=lw, label="LS")
plt.plot(xx, np.where(np.abs(xx) < t , xx, t*xx/np.abs(xx)), color='blue', lw=lw, label="Huber")
plt.plot(xx, np.where(np.abs(xx) < t , xx*(1 - np.square(xx/t)), 0), color='green', lw=lw, label="Bisquare")

ax1 = plt.gca()
ax1.set_xlabel('Error e')
ax1.set_ylabel(r'$\psi(e)$')


plt.subplot(223)
plt.plot(xx, np.ones_like(xx), color='red', lw=lw, label="LS")
plt.plot(xx, np.where(np.abs(xx) < t , np.ones_like(xx), t/np.abs(xx)), color='blue', lw=lw, label="Huber")
plt.plot(xx, np.where(np.abs(xx) < t , (1 - np.square(xx/t)), 0), color='green', lw=lw, label="Bisquare")

ax1 = plt.gca()


ax1.set_xlabel('Error e')
ax1.set_ylabel('w(e)')

ax1.legend()

plt.savefig('MEstimatorFunction.png',dpi=300)
