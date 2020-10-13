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
from matplotlib import cm
sns.set()

SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 26
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
sample_rate, audio = read(source)

frame_size = 0.025 # 25ms
frame_stride = 0.01 # 10ms

frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
signal_length = len(audio)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))
num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame
# num_frames 5938
pad_signal_length = num_frames * frame_step + frame_length
z = np.zeros((pad_signal_length - signal_length))
pad_signal = np.append(audio, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T

frames = pad_signal[indices.astype(np.int32, copy=False)]

# get hamming function
hammingFunc = np.hamming(frame_length)
plt.figure(1, figsize=(17, 8))
plt.subplot(121)
plt.plot(hammingFunc, lw=2)
plt.xlabel(r'Index $n$')
plt.ylabel(r'$h$')
plt.subplot(122)
plt.plot(frames[0], '-.', label='original', lw=2)
plt.plot(frames[0] * hammingFunc, label='windowed', lw=2)
plt.xlabel(r'Index $n$')
plt.ylabel(r'Amplitude')
plt.legend()
plt.savefig('hammingFuncDemo.png', dpi=300)

# now do the windowing using hamming function
frames *= np.hamming(frame_length)

# Fourier transform and power spectrum
# Compute the one-dimensional discrete Fourier Transform for real input.
NFFT = 512
fft_res = np.fft.rfft(frames, NFFT)
mag_frames = np.absolute(fft_res)  # Magnitude of the FFT, mag_frames has NFFT/2 + 1 frequency components
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum


plt.figure(2, figsize=(10, 7))
plt.xlabel(r'Frequency $k$')
plt.ylabel(r'FFT Magnitude $|S(k)|$')
plt.plot(mag_frames[0],lw=2, label='FFT zero-padding')
mag_noPad = np.abs(np.fft.rfft(frames[0], len(frames[0])))
plt.plot(mag_noPad,lw=2, label='DFT no padding')
plt.legend()
plt.savefig('FFTDemo.png', dpi=300)

sns.set_style("whitegrid", {'axes.grid' : False})
plt.figure(3, figsize=(10, 7))
plt.imshow(mag_frames[:500].T, cmap='jet', origin='lower')
plt.show()
plt.xlabel('Frame Index')
plt.ylabel('FFT Magnitude')
plt.savefig('FFTMagPlot.png',dpi=300)

plt.figure(4, figsize=(8, 7))
plt.imshow(pow_frames[:300].T, cmap='jet', origin='lower')
plt.show()
plt.xlabel('Frame Index')
plt.ylabel('Power')
plt.savefig('SignalPowerPlot.png',dpi=300)