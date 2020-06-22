from misc.constants import freq
from misc.datasets import datasets
from postprocessing.smoothing import *
import numpy as np
import misc


def postprocessing(results, scalers, dataset):
    """
    Do the whole post-processing pipeline by:
    1. rescaling the results
    2. formatting the results
    :param results:
    :param scalers:
    :return:
    """
    results = _rescale(results, scalers)
    results = _reshape(results, max([freq,datasets[dataset]["glucose_freq"]]))

    return results

def postprocessing_all_iter(results, scalers, dataset):
    """
    Do the whole post-processing pipeline by:
    1. rescaling the results
    2. formatting the results
    :param results:
    :param scalers:
    :return:
    """

    scaled_results = []
    for res_cv, scaler in zip(results, scalers):
        mean = scaler.mean_[-1]
        std = scaler.scale_[-1]
        scaled_results_cv = []
        for res_iter in res_cv:
            scaled_results_cv.append(res_iter * std + mean)
        scaled_results.append(scaled_results_cv)

    freq_ds = max([freq,datasets[dataset]["glucose_freq"]])

    results = [[res_iter.resample(str(freq_ds) + "min").mean() for res_iter in res_cv] for res_cv in scaled_results]

    return results

def _rescale(results, scalers):
    """
    Before evaluating the results we need to rescale the glucose predictions that have been standardized.
    :param results: array of shape (cv_fold, n_predictions, 2);
    :param scalers: array of scalers used to standardize the data;
    :return: rescaled results;
    """
    scaled_results = []
    for res, scaler in zip(results, scalers):
        mean = scaler.mean_[-1]
        std = scaler.scale_[-1]
        scaled_results.append(res * std + mean)

    return scaled_results

def _smooth(results, smoothing_params):
    return smoothing_params["func"](results, smoothing_params["params"])


def _reshape(results, freq):
    """
    Reshape (resample) the results into the given sampling frequency
    :param results: array of dataframes with predictions and ground truths
    :param freq: sampling frequency
    :return: reshaped results
    """
    return [res_.resample(str(freq) + "min").mean() for res_ in results]
