import numpy as np
import scipy.signal


def get_butterworth_filter(f, cutoff, btype="low", order=2):
    ba = scipy.signal.butter(N=order, Wn=np.divide(cutoff, f/2.), btype=btype)
    return DigitalFilter(ba[0], ba[1])


class DigitalFilter:

    def __init__(self, b, a):
        self._bs = b
        self._as = a
        self._xs = [0]*len(b)
        self._ys = [0]*(len(a)-1)

    def process(self, x):
        if np.isnan(x):  # ignore nans, and return as is
            return x

        self._xs.insert(0, x)
        self._xs.pop()
        y = (np.dot(self._bs, self._xs) / self._as[0]
             - np.dot(self._as[1:], self._ys))
        self._ys.insert(0, y)
        self._ys.pop()
        return y

    def __call__(self, x):
        return self.process(x)
