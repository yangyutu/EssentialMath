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


xmin, xmax = -5, 5
xx = np.linspace(xmin, xmax, 100)
lw = 2
plt.figure(1, figsize = (14,14))
plt.subplot(221)
plt.plot(xx, xx**2, color='gold', lw=lw, label="MSE")
ax1 = plt.gca()
#ax1.set_xlabel('Prediction Error')
ax1.set_ylabel('Loss')
ax1.set_title("MSE Loss")
plt.subplot(222)
plt.plot(xx, np.abs(xx), color='teal', lw=lw, label="MAE")
ax1 = plt.gca()
#ax1.set_xlabel('Prediction Error')
ax1.set_ylabel('Loss')
ax1.set_title("MAE Loss")

def sm_mae(pred, delta):
    """
    true: array of true values    
    pred: array of predicted values
    
    returns: smoothed mean absolute error loss
    """
    loss = np.where(np.abs(pred) < delta , 0.5*((pred)**2), delta*np.abs(pred) - 0.5*(delta**2))
    return np.sum(loss)

plt.subplot(223)
ax1 = plt.gca()

delta = [0.1, 1, 3]

losses_huber = [[sm_mae(xx[i], q) for i in range(len(xx))] for q in delta]

# plot 
for i in range(len(delta)):
    ax1.plot(xx, losses_huber[i], label = r'$\delta=$' + str(delta[i]))
ax1.set_xlabel('Prediction Error')
ax1.set_ylabel('Loss')
ax1.set_title("Huber Loss")
ax1.legend()
ax1.set_ylim(bottom=-1, top = 15)


def logcosh( pred):
    loss = np.log(np.cosh(pred))
    return np.sum(loss)


plt.subplot(224)
ax1 = plt.gca()



loss_logcosh = [logcosh(xx[i]) for i in range(len(xx))]

# plot 
ax1.plot(xx, loss_logcosh)
ax1.set_xlabel('Prediction Error')
ax1.set_ylabel('Loss')
ax1.set_title("Log-Cosh Loss")
plt.show()