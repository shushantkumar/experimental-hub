import numpy as np
import scipy.signal

def bpm_from_inds(inds, ts):
    if len(inds) < 2:
        return np.nan

    return 60. / np.mean(np.diff(ts[inds]))

def get_sampling_rate(ts):
    return 1. / np.mean(np.diff(ts))

def from_peaks(vs, ts, mindist=0.35):
    if len(ts) != len(vs) or len(ts) < 2:
        return np.nan
    f = get_sampling_rate(ts)
    peaks, _ = scipy.signal.find_peaks(vs, distance=int(f*mindist))

    return bpm_from_inds(peaks, ts)

def from_fft(vs, ts):
    f = get_sampling_rate(ts)
    vf = np.fft.fft(vs)
    xf = np.linspace(0.0, f/2., len(vs)//2)
    return 60 * xf[np.argmax(np.abs(vf[:len(vf)//2]))]

class HRCalculator:
    def __init__(self, update_interval=30, winsize=300, filt_fun=None, hr_fun=None):
        self._counter = 0
        self.update_interval = update_interval
        self.winsize = winsize
        self.filt_fun = filt_fun
        self.hr_fun = hr_fun or from_peaks
        self.updated_hr = 0

    def update(self, rppg):
        self._counter += 1
        if self._counter >= self.update_interval:
            self._counter = 0
            ts = rppg.get_ts(self.winsize)
            vs = next(rppg.get_vs(self.winsize))
            if self.filt_fun is not None and callable(self.filt_fun):
                vs = self.filt_fun(vs)
            hr = self.hr_fun(vs, ts)
            self.updated_hr = hr
