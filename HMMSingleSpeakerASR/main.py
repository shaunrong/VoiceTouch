#!/usr/bin/env python
import os
from scipy.io import wavfile
import numpy as np
import scipy
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided
from GMMHMM import gmmhmm
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix

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
    print 'Words spoken: {}'.format(spoken)

    data = np.zeros((len(fpaths), 32000))
    maxsize = -1
    for n, file in enumerate(fpaths):
        _, d = wavfile.read(file)
        data[n, :d.shape[0]] = d
        if d.shape[0] > maxsize:
            maxsize = d.shape[0]
    data = data[:, :maxsize]
    print "Number of files total: {}".format(data.shape[0])

    all_labels = np.zeros(data.shape[0])
    for n, l in enumerate(set(labels)):
        all_labels[np.array([i for i, _ in enumerate(labels) if _ == l])] = n
    print "Labels and label indices {}".format(all_labels)

    # Below is feature extraction, using simple peak detection
    plt.figure()
    plt.plot(data[0, :], color='steelblue')
    plt.title('Timeseries example for {}'.format(labels[0]))
    plt.xlim(0, 3500)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude (signed 16 bit)')

    plt.figure()
    log_freq = 20 * np.log(np.abs(stft(data[0, :])) + 1)
    plt.imshow(log_freq, cmap='gray', interpolation=None)
    plt.xlabel("Freq (bin)")
    plt.ylabel("Time (overlapped frames)")
    plt.ylim(log_freq.shape[1])
    plt.title("PSD of {} example".format(labels[0]))
    plt.show()

    plot_data = np.abs(stft(data[20, :]))[15, :]
    values, locs = peakfind(plot_data, n_peaks=6)
    fp = locs[values > -1]
    fv = values[values > -1]
    plt.figure()
    plt.plot(plot_data, color='steelblue')
    plt.plot(fp, fv, 'x', color='darkred')
    plt.xlabel('Peak location example')
    plt.ylabel('Amplitude')
    plt.show()

    all_obs = []
    for i in range(data.shape[0]):
        d = np.abs(stft(data[i, :]))
        n_dim = 6
        obs = np.zeros((n_dim, d.shape[0]))
        for r in range(d.shape[0]):
            _, t = peakfind(d[r, :], n_peaks=n_dim)
        if i % 10 == 0:
            print "Processed obs {}".format(i)
        all_obs.append(obs)


    sss = StratifiedShuffleSplit(all_labels, test_size=0.1, random_state=0)

    for n, i in enumerate(all_obs):
        all_obs[n] /= all_obs[n].sum(axis=0)

    for train_index, test_index in sss:
        X_train, X_test = all_obs[train_index], all_obs[test_index]
        y_train, y_test = all_labels[train_index], all_labels[test_index]
    print 'Size of training matrix: {}'.format(X_train.shape)
    print 'Size of testing matrix: {}'.format(X_test.shape)

    ys = set(all_labels)
    ms = [gmmhmm(6) for y in ys]
    _ = [m.fit(X_train[y_train == y, :, :]) for m, y in zip(ms, ys)]
    ps = [m.transform(X_test) for m in ms]
    res = np.vstack(ps)
    predicted_labels = np.argmax(res, axis=0)
    missed = (predicted_labels != y_test)
    print 'Test accuracy: {} percent'.format((100 * (1 - np.mean(missed))))

    plt.figure()
    cm = confusion_matrix(y_test, predicted_labels)
    plt.matshow(cm, cmap='gray')
    ax = plt.gca()
    _ = ax.set_xticklabels([" "] + [l[:2] for l in spoken])
    _ = ax.set_yticklabels([" "] + spoken)
    plt.title('Confusion matrix, single speaker')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def stft(x, fftsize=64, overlap_pct=0.5):
    hop = int(fftsize * (1 - overlap_pct))
    w = scipy.hanning(fftsize + 1)[:-1]
    raw = np.array([np.fft.rfft(w * x[i:i + fftsize]) for i in range(0, len(x) - fftsize, hop)])
    return raw[:, :(fftsize / 2)]


def peakfind(x, n_peaks, l_size=3, r_size=3, c_size=3, f=np.mean):
    win_size = l_size + r_size + c_size
    shape = x.shape[:-1] + (x.shape[-1] - win_size + 1, win_size)
    strides = x.strides + (x.strides[-1], )
    xs = as_strided(x, shape=shape, strides=strides)

    def is_peak(x):
        centered = (np.argmax(x) == l_size + int(c_size / 2))
        l = x[:l_size]
        c = x[l_size:l_size + c_size]
        r = x[-r_size:]
        passes = np.max(c) > np.max([f(l), f(r)])
        if centered and passes:
            return np.max(c)
        else:
            return -1

    r = np.apply_along_axis(is_peak, 1, xs)
    top = np.argsort(r, None)[::-1]
    heights = r[top[:n_peaks]]
    top[top > -1] = top[top > -1] + l_size + int(c_size / 2.)
    return heights, top[:n_peaks]


if __name__ == '__main__':
    main()
