#!/usr/bin/env python
import os
from scipy.io import wavfile
import numpy as np
import scipy
import matplotlib.pyplot as plt

__author__ = 'Shaun Rong'
__version__ = '0.1'
__maintainer__ = 'Shaun Rong'
__email__ = 'rongzq08@gmail.com'


def main():

    fpaths = []
    labels = []
    spoken = []
    for f in os.listdir('audio'):
        for w in os.listdir('audio/{}'.format(f)):
            fpaths.append('audio/{}/{}'.format(f, w))
            labels.append(f)
            if f not in spoken:
                spoken.append(f)

    data = np.zeros((len(fpaths), 32000))
    maxsize = -1
    for n, file in enumerate(fpaths):
        _, d = wavfile.read(file)
        data[n, :d.shape[0]] = d
        if d.shape[0] > maxsize:
            maxsize = d.shape[0]
    data = data[:, :maxsize]
    all_labels = np.zeros(data.shape[0])
    for n, l in enumerate(set(labels)):
        all_labels[np.array([i for i, _ in enumerate(labels) if _ == l])] = n

    plt.plot(data[0, :], color='steelblue')
    plt.title('Timeseries example for {}'.format(labels[0]))
    plt.xlim(0, 3500)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude (signed 16 bit)')
    plt.figure()

    log_freq = 20 * np.log(np.abs(stft(data[0, :])))


def stft(x, fftsize=64, overlap_pct=0.5):
    hop = int(fftsize * (1 - overlap_pct))
    w = scipy.hanning(fftsize + 1)[:-1]
    raw = np.array([np.fft.rfft(w * x[i:i + fftsize]) for i in range(0, len(x) - fftsize, hop)])
    return raw[:, :(fftsize / 2)]

if __name__ == '__main__':
    main()
