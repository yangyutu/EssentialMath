# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 20:02:36 2020

@author: yangy
"""


import librosa
import matplotlib.pyplot as plt
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
plt.subplots_adjust(wspace=0.2, hspace =0.2)
plt.close('all')


y1, sr = librosa.load(r'G:\Download\Yang.mp3', sr=16000)
y2, sr = librosa.load(r'G:\Download\Ruoyi.mp3', sr=16000)

frame_size = 0.025 # 25ms
frame_stride = 0.01 # 10ms

frame_length, frame_step = frame_size * sr, frame_stride * sr  # Convert from seconds to samples
signal_length = len(y1)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))

plt.figure(1, figsize=(14, 4))
plt.plot(np.arange(len(y1)) / sr, y1, label='male')
plt.plot(np.arange(len(y2)) / sr, y2, label='female')
plt.legend()
plt.xlabel('Time, s')
plt.ylabel('Amptitude')
plt.subplots_adjust(bottom = 0.2)
plt.savefig('wavplot.png', dpi=300)




## compute mel spectrogram

# perform melspectrogram here is equivalent to two steps
# Compute power spectrum
# S = np.abs(librosa.stft(y, n_fft))**2
# Build a Mel filter
# mel_basis = filters.mel(sr, n_fft)
# mel = np.dot(mel_basis, S)
num_mels = 80
n_fft = 512

melSpec1 = librosa.feature.melspectrogram(y=y1, sr=sr, n_mels=num_mels, n_fft=n_fft, win_length=frame_length, hop_length=frame_step)
melSpec2 = librosa.feature.melspectrogram(y=y2, sr=sr, n_mels=num_mels, n_fft=n_fft, win_length=frame_length, hop_length=frame_step)

melSpec1 = (melSpec1 - melSpec1.mean(axis=1, keepdims=True)) / (melSpec1.std(axis=1, keepdims=True)+1e-16)
melSpec2 = (melSpec2 - melSpec1.mean(axis=1, keepdims=True)) / (melSpec2.std(axis=1, keepdims=True)+1e-16)
sns.set_style("whitegrid", {'axes.grid' : False})
plt.figure(2, figsize=(14, 4))
plt.subplot(1, 2, 1)
plt.imshow(melSpec1, cmap='jet', aspect='auto', origin='lower')
plt.xlabel('Frame Index')
plt.ylabel('Mel Frequency')
plt.subplots_adjust(bottom = 0.2)
plt.subplot(1, 2, 2)
plt.imshow(melSpec2, cmap='jet', aspect='auto', origin='lower')
plt.xlabel('Frame Index')
plt.ylabel('Mel Frequency')
plt.subplots_adjust(bottom = 0.2)
plt.savefig('Melspectrogram.png', dpi=300)

## compute MFCC 
num_mfcc = 13

mfcc1 = librosa.feature.mfcc(y=y1, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft,  win_length=frame_length, hop_length=frame_step)
mfcc2 = librosa.feature.mfcc(y=y2, sr=sr, n_mfcc=num_mfcc, n_fft=n_fft,  win_length=frame_length, hop_length=frame_step)
mfcc1 = (mfcc1 - mfcc1.mean(axis=1, keepdims=True)) / (mfcc1.std(axis=1, keepdims=True)+1e-16)
mfcc2 = (mfcc2 - mfcc2.mean(axis=1, keepdims=True)) / (mfcc2.std(axis=1, keepdims=True)+1e-16)

plt.figure(3, figsize=(14, 4))
plt.subplot(1, 2, 1)
plt.imshow(mfcc1, cmap='jet', aspect='auto', origin='lower')
plt.xlabel('Frame Index')
plt.ylabel('MFCC')
plt.subplots_adjust(bottom = 0.2)
plt.subplot(1, 2, 2)
plt.imshow(mfcc2, cmap='jet', aspect='auto', origin='lower')
plt.xlabel('Frame Index')
plt.ylabel('MFCC')
plt.subplots_adjust(bottom = 0.2)
plt.savefig('MFCC.png', dpi=300)
