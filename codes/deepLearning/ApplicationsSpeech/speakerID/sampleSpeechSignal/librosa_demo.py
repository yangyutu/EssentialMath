# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 20:35:46 2020

@author: yangy
"""

import librosa
import matplotlib.pyplot as plt
import numpy as np


source = 'male/male1.wav'
y, sample_rate = librosa.load(source, sr=16000)

frame_size = 0.025 # 25ms
frame_stride = 0.01 # 10ms

frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
signal_length = len(y)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))
num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame


# get short-term FFT result
# here in stft hann window function is automatically applied
# res has shape (1 + n_fft/2, frames)
fft_result = librosa.stft(y, n_fft=512, hop_length=frame_step, win_length=frame_length)





# get Mel filter banks
# norm flag: whether to normalize by triangle area.
n_mels = 10
filters = librosa.filters.mel(sr=16000, n_fft=512, n_mels=n_mels, norm=1)

for i in range(n_mels):
    plt.plot(filters[i], lw=2)

