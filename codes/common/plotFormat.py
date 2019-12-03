# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 17:30:54 2018

@author: yuguangyang
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def set_pubAll():
    matplotlib.rcParams.update({'font.size': 14})
    matplotlib.rc('axes', linewidth=2)
    matplotlib.rcParams['xtick.major.size'] = 6
    matplotlib.rcParams['xtick.major.width'] = 2
    matplotlib.rcParams['xtick.minor.size'] = 3
    matplotlib.rcParams['xtick.minor.width'] = 1
    matplotlib.rcParams['xtick.minor.visible'] = False
    matplotlib.rcParams['xtick.direction'] = 'in'
    matplotlib.rcParams['font.weight'] = 'bold'
    matplotlib.rcParams['ytick.major.size'] = 6
    matplotlib.rcParams['ytick.major.width'] = 2
    matplotlib.rcParams['ytick.minor.size'] = 3
    matplotlib.rcParams['ytick.minor.width'] = 1
    matplotlib.rcParams['ytick.minor.visible'] = False
    matplotlib.rcParams['ytick.direction'] = 'in'
    matplotlib.rcParams['legend.fontsize'] = 'small'
    matplotlib.rcParams.update({'figure.autolayout': True})
def set_pub():
#    plt.rc('font', weight='bold')    # bold fonts are easier to see
    plt.rc('xtick', labelsize=12)     # tick labels bigger
    plt.rc('ytick', labelsize=12)     # tick labels bigger
    plt.rc('lines', lw=2, color='k') # thicker black lines (no budget for color!)
    #rc('grid', c='0.5', ls='-', lw=0.5)  # solid gray grid lines
    plt.rc('savefig', dpi=300)       # higher res outputs
    
def polish(ax):
    plt.tight_layout()
    ax.set_aspect(1./ax.get_data_ratio())
    ax.minorticks_on()
    # adjusting linewidth of axes
    # https://stackoverflow.com/questions/2553521/setting-axes-linewidth-without-changing-the-rcparams-global-dict
    for axis in ['top','bottom','left','right']: 
      ax.spines[axis].set_linewidth(2)
    # We change the fontsize of minor ticks label 
    ax.tick_params(axis='both',which='both',top=True,right=True,direction='in')
    ax.tick_params(axis='both', which='major',length=10,width=2)
    ax.tick_params(axis='both', which='minor',length=6,width=2)    


if __name__=='__main__':
    set_pub()
    plt.close('all')
    x = np.linspace(0,10,100)
    y = np.cos(x)
    z = np.sin(x)
    
    fig1 = plt.figure(1)
    plt.plot(x,y,linewidth=2)
    ax = plt.axes()
    
    ax.set_xlabel('x',fontsize=20,fontweight='bold')
    ax.set_ylabel('y',fontsize=20,fontweight='bold')
    
    polish(ax)
    plt.savefig('demo.pdf')
    
    
    #ax = fig.add_subplot(111,aspect='equal')
    #ax = fig.add_subplot(111,aspect=1.0)
    #ax.set_aspect('equal')
    #plt.axes().set_aspect('equal')
    
    
    #fig2 = plt.figure(3)
    #ax1 = fig2.add_axes()
    #ax1.plot(x,y)
    
    #fig, ax = plt.subplots()
    fig = plt.figure(10,figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.plot(x,y)
    ax.plot(x,z,marker='o')
    ax.set(xlim=[0,10],ylim=[-1,1])
    #ax.set(title="An example axes",ylabel='Y_label',xlabel='X_label')
    ax.set_xlabel('$y=sin(x)$',fontsize=20,fontweight='bold')
    ax.set_ylabel('ylabel',fontsize=20,fontweight='bold')
    polish(ax)
    
    
    plt.savefig('demo2.pdf')
    
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ax.plot([1, 2, 3, 4], [10, 20, 25, 30], color='lightblue', linewidth=3)
    ax.scatter([0.3, 3.8, 1.2, 2.5], [11, 25, 9, 26], color='darkgreen', marker='^')
    ax.set_xlim(0.5, 4.5)
    polish(ax)
    plt.show()
