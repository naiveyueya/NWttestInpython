# -*- coding:utf-8 -*-
#@author: naiveyueya
__version__ = '0.0.1'
import numpy as np
from collections import namedtuple
from scipy.stats import distributions


def _ttest_finish(df, t):#from scipy.stats
    prob = distributions.t.sf(np.abs(t), df) * 2  # use np.abs to get upper tail
    if t.ndim == 0:
        t = t[()]

    return t, prob

NWt_1sampleResult = namedtuple('NWT_1sampResult', ('statistic', 'pvalue'))
def nwttest_1samp(a, popmean, axis=0,L='auto'):
    a = np.array(a)
    N = len(a)
    if L=='auto':
        L = 4 * (N/100) ** (2 / 9)
    elif type(L) != int:
        print('Error Input Lag')
        return
    df = N-1
    e = a - np.mean(a)
    residuals = np.sum(e**2)
    Q = 0
    for i in range(L):
        w_l = 1 - (i+1)/(1+L)
        for j in range(1,N):
            Q += w_l*e[j]*e[j-(i+1)]
    S = residuals + 2*Q
    nw_var = S/N
    d = np.mean(a,axis) - popmean
    nw_sd = np.sqrt(nw_var / float(df))
    with np.errstate(divide='ignore', invalid='ignore'):
        t = np.divide(d, nw_sd)
    t,prob = _ttest_finish(df,t)

    return NWt_1sampleResult(t,prob)
