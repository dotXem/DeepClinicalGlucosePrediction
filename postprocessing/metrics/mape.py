import numpy as np

def MAPE(results):
    """
        Compute the mean absolute percentage error  of the predictions, with is a normalized mean absolute error, expressed in %
        :param results: dataframe with predictions and ground truths
        :return: fitness
    """
    results = results.loc[(results != 0).all(axis=1)] # drop zeros because we can't divide by zeros - zeros are erroneous values anyway
    y_true, y_pred = results.dropna().values.transpose()
    return 100 * np.nanmean(np.abs((y_true-y_pred)/y_true))