# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 14:15:29 2020

@author: yangy

https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html
"""
import os
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
sns.set()
SMALL_SIZE = 22
MEDIUM_SIZE = 25
BIGGER_SIZE = 30
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.subplots_adjust(wspace =0.5, hspace =0.5)
plt.margins(0.3)
#plt.subplots_adjust(left=0.5, bottom=0.1)
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

pad_signal_length = num_frames * frame_step + frame_length
z = np.zeros((pad_signal_length - signal_length))
pad_signal = np.append(audio, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[indices.astype(np.int32, copy=False)]

# now do the windowing using hamming function
frames *= np.hamming(frame_length)

# Fourier transform and power spectrum
# Compute the one-dimensional discrete Fourier Transform for real input.
NFFT = 512
mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

nfilt = 16
NFFT = 512
low_freq_mel = 0
high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
bin = np.floor((NFFT + 1) * hz_points / sample_rate)

fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])   # left
    f_m = int(bin[m])             # center
    f_m_plus = int(bin[m + 1])    # right

    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
filter_banks = np.dot(pow_frames, fbank.T)
filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
filter_banks = 20 * np.log10(filter_banks)  # dB
plt.figure(1, figsize=(8, 7))
plt.plot(filter_banks[0])
plt.xlabel('Frequency')
plt.ylabel('power, dB')
plt.subplots_adjust(left=0.15)


num_ceps = 12
from scipy.fftpack import dct
mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] 
(nframes, ncoeff) = mfcc.shape
n = np.arange(ncoeff)
cep_lifter = num_ceps * 2
lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
mfcc *= lift  #*
plt.figure(2)
plt.plot(lift)

sns.set_style("whitegrid", {'axes.grid' : False})
plt.figure(3, figsize=(16, 7))
plt.imshow(mfcc[:2500].T, aspect='auto',cmap='jet', origin='lower')
plt.show()
plt.xlabel('Frame Index')
plt.ylabel('MFCC')
plt.savefig('MFCC.png', dpi=300)

# to balance the spectrum and improve the Signal-to-Noise (SNR), we can simply subtract the mean of each coefficient from all frames.
filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)