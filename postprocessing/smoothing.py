import pandas as pd
import numpy as np

def smooth_results(results, params):
    results = results.copy()
    results.y_pred = params["func"](results.y_pred, *params["params"]).astype(np.float32)
    return results

def moving_average(x, N, wet_ratio=1.0):
    """
        Apply a moving average smoothing to the input data
        :param x: input time-series
        :param N: wideness of the moving avergae window (from 1 to 5)
        :return: smoothed time-series
    """
    ma = pd.Series(x).rolling(window=N).mean()
    ma[pd.isna(ma)] = x[pd.isna(ma)]
    return wet_ratio * ma.values + (1 - wet_ratio) * x.values


def exponential_smoothing(x, alpha):
    """
        Apply an exponential smoothing to the input data
        :param x: input time-series
        :param alpha: smoothness (between 0 and 1) coefficient
        :return: smoothed time-series
    """
    es = pd.DataFrame(x).ewm(alpha=alpha).mean()
    return es

