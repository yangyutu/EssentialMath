# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 12:03:13 2020

@author: yangy
"""

import os
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
sns.set()
SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 25
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.subplots_adjust(wspace =0.5, hspace =0.5)
plt.close('all')



source = 'male/male1.wav'
sr, audio = read(source)
x = np.arange(len(audio)) / sr
plt.figure(1, figsize = (15, 12))
plt.subplot(211)
plt.plot(x, audio)
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude')

start = 10000
end = 15000
sample = audio[start:end]
x = np.arange(start, end) / sr * 1000
plt.subplot(212)
plt.plot(x, sample)
plt.xlabel('Time (msec)')
plt.ylabel('Amplitude')

plt.savefig('audioSignalExample.png', dpi=300)


sns.set_style("whitegrid", {'axes.grid' : False})

plt.figure(5, figsize = (15, 12))
start = 25000
end = 45000
sample = audio[start:end]
x = np.arange(start, end) / sr * 1000
plt.subplot(211)
plt.plot(x, sample,color='black')
plt.subplot(212)
plt.plot(x, sample,color='red')
plt.xlabel('Time (sec)')
plt.ylabel('Amplitude')
plt.savefig('audioSignalExampleWhite.png', dpi=300)