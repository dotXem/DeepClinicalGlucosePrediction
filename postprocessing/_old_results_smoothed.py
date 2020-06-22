# from pathlib import Path
#
#
# import misc.constants
# import misc.datasets
# from misc import constants as cs
#
#
# import numpy as np
# import os
# import pandas as pd
# from statsmodels.tsa.api import SimpleExpSmoothing
# import misc
# from postprocessing.results import ResultsSubject, ResultsDataset
# import matplotlib.pyplot as plt
#
# class ResultsSubjectSmoothed(ResultsSubject):
#     def compute_smooth_results(self, window_size=2, wet_ratio=1.0):
#         results_smoothed = []
#         for results_split in self.results:
#             results_smoothed_split = results_split.copy()
#             # results_smoothed_split.y_pred = self.moving_average(results_smoothed_split.y_pred, window_size, wet_ratio).astype(np.float32)
#             results_smoothed_split.y_pred = self.exponential_smoothing(results_smoothed_split.y_pred, wet_ratio).astype(np.float32)
#             results_smoothed.append(results_smoothed_split)
#         return ResultsSubject(self.model, self.experiment + "_smoothed_win"+str(window_size)+ "_wet"+str(wet_ratio),self.ph, self.dataset, self.subject,self.params, results_smoothed).compute_mean_std_results()
#
#     def get_all_smoothing_results(self, metric):
#         wet_ratios = np.arange(0,10,1) / 10
#         res = []
#         for window_size in range(10):
#             window_res = []
#             for wet_ratio in wet_ratios:
#                 window_res.append(self.compute_smooth_results(window_size, wet_ratio)[0][metric])
#             res.append(window_res)
#         return np.reshape(res,(-1,1))
#
#     def plot_smoothing_evolution(self, metric):
#         wet_ratios = np.arange(0,10,1) / 10
#         plt.figure()
#         res = []
#         for window_size in range(10):
#             window_res = []
#             for wet_ratio in wet_ratios:
#                 window_res.append(self.compute_smooth_results(window_size, wet_ratio)[0][metric])
#             plt.plot(wet_ratios,window_res,label="window="+str(window_size))
#             res.append(window_res)
#         return res
#
#     def moving_average(self, x, N, wet_ratio=1.0):
#         """
#             Apply a moving average smoothing to the input data
#             :param x: input time-series
#             :param N: wideness of the moving avergae window (from 1 to 5)
#             :return: smoothed time-series
#         """
#         ma = pd.Series(x).rolling(window=N).mean()
#         ma[pd.isna(ma)] = x[pd.isna(ma)]
#         return wet_ratio * ma.values + (1-wet_ratio) * x.values
#
#     def exponential_smoothing(self, x, alpha):
#         """
#             Apply an exponential smoothing to the input data
#             :param x: input time-series
#             :param alpha: smoothness (between 0 and 1) coefficient
#             :return: smoothed time-series
#         """
#         # return SimpleExpSmoothing(x).fit(alpha).fittedvalues
#         ma = pd.DataFrame(x).ewm(alpha=alpha).mean()
#         return ma
#
#
# class ResultsDatasetSmoothed(ResultsDataset):
#     def compute_smoothed_results(self, window_size=2, wet_ratio=1.0, details=False):
#         """
#         Loop through the subjects of the dataset, and compute the mean performances
#         :return: mean of metrics, std of metrics
#         """
#         res = []
#         for subject in self.subjects:
#             res_subject = ResultsSubjectSmoothed(self.model, self.experiment, self.ph, self.dataset, subject,
#                                          legacy=self.legacy).compute_smooth_results(window_size, wet_ratio)
#             if details:
#                 print(self.dataset, subject, res_subject)
#
#             res.append(res_subject[0])  # only the mean
#
#         keys = list(res[0].keys())
#         res = [list(res_.values()) for res_ in res]
#         mean, std = np.nanmean(res, axis=0), np.nanstd(res, axis=0)
#         return dict(zip(keys, mean)), dict(zip(keys, std))
#
#     def compute_all_smoothed_results(self):
#         wet_ratios = np.arange(0,10,1) / 10
#         res = []
#         for window_size in [7]: #range(10):
#             window_res = []
#             for wet_ratio in wet_ratios:
#                 window_res.append(self.compute_smoothed_results(window_size,wet_ratio)[0])
#             res.append(window_res)
#         return np.reshape(res,(-1))
#
# def compare_with_twostep(self, model, dataset, exp_smoothed, exp_twostep, metric1, metric2):
#     from postprocessing.results_twostep import ResultsDatasetTwoStep
#     res_smoothed = ResultsDatasetSmoothed(model,exp_smoothed,30,dataset).compute_all_smoothed_results()
#     res_smoothed2 = ResultsDatasetSmoothed(model,"step1_test_mixed_hypo",30,dataset).compute_all_smoothed_results()
#     res_twostep = ResultsDatasetTwoStep(model,exp_twostep,30,dataset).compute_results_all_iter()
#
#     metric1_twostep = [res_twostep_i[metric1] for res_twostep_i in res_twostep]
#     metric2_twostep = [res_twostep_i[metric2] for res_twostep_i in res_twostep]
#     metric1_smoothed = [res_smoothed_i[metric1] for res_smoothed_i in res_smoothed]
#     metric2_smoothed = [res_smoothed_i[metric2] for res_smoothed_i in res_smoothed]
#     metric1_smoothed2 = [res_smoothed_i[metric1] for res_smoothed_i in res_smoothed2]
#     metric2_smoothed2 = [res_smoothed_i[metric2] for res_smoothed_i in res_smoothed2]
#
#     plt.figure()
#     plt.plot(metric1_twostep, metric2_twostep, "bo")
#     plt.plot(metric1_smoothed, metric2_smoothed, "rx")
#     plt.plot(metric1_smoothed2, metric2_smoothed2, "gd")
#
# # def smooth_twostep(model, dataset, exp_twostep, metric1, metric2):
# #     from postprocessing.results_twostep import ResultsDatasetTwoStep
# #     res =
#
# class ResultsSubjectTwoStepSmoothed():
#     def __init__(self, model, experiment, ph, dataset, subject, params=None, results=None):
#
#         self.model = model
#         self.experiment = experiment
#         self.ph = ph
#         self.dataset = dataset
#         self.subject = subject
#         self.freq = np.max([misc.constants.freq, misc.datasets.datasets[dataset]["glucose_freq"]])
#
#         if results is None and params is None:
#             self.params, self.results = self.load_raw_results()
#         else:
#             self.results = results
#             self.params = params
#
#     def load_raw_results(self, legacy=False):
#         """
#         Load the results from previous instance of ResultsSubject that has saved the them
#         :param legacy: if legacy object shall  be used
#         :return: (params dictionary), dataframe with ground truths and predictions
#         """
#         file = self.dataset + "_" + self.subject + ".npy"
#         path = os.path.join(cs.path, "results", self.model, self.experiment, "ph-" + str(self.ph), file)
#
#         params, results = np.load(path, allow_pickle=True)
#         dfs = []
#         for results_cv in results:
#             dfs_cv = []
#             for results_iter in results_cv:
#                 df = pd.DataFrame(results_iter, columns=["datetime", "y_true", "y_pred"])
#                 df = df.set_index("datetime")
#                 df = df.astype("float32")
#                 dfs_cv.append(df)
#             dfs.append(dfs_cv)
#         return params, dfs
#
#     def save_raw_results(self):
#         """
#         Save the results and params
#         :return:
#         """
#         dir = os.path.join(cs.path, "results", self.model, self.experiment, "ph-" + str(self.ph))
#         Path(dir).mkdir(parents=True, exist_ok=True)
#
#         saveable_results = np.array([[res_.reset_index().to_numpy() for res_ in res] for res in self.results])
#
#         np.save(os.path.join(dir, self.dataset + "_" + self.subject + ".npy"), [self.params, saveable_results])
#
#     def compute_results_iter_smooth(self, iter=0, window_size=1, ratio=1):
#         res = [res_[iter] for res_ in self.results]
#         results_subject = ResultsSubjectSmoothed(self.model, self.experiment, self.ph, self.dataset, self.subject, self.params,
#                                          res)
#         return results_subject.compute_smooth_results(window_size, ratio)
#
#     def compute_results_all_iter_smooth(self, window_size, ratio):
#         res_iter = []
#         for iter in range(len(self.results[0])):
#             res_iter.append(self.compute_results_iter_smooth(iter, window_size, ratio)[0])
#         return res_iter
#
#     def compute_evolution(self):
#         dict_arr = []
#         for iter in range(len(self.results[0])):
#             dict_arr.append(self.compute_results_iter(iter)[0])
#
#         evolution = {}
#         for key in dict_arr[0].keys():
#             evolution[key] = [d[key] for d in dict_arr]
#
#         return evolution
#
#     def compute_mean_std_results(self, split_by_day=False):
#         """
#         From the raw metrics scores, compute the mean and std
#         :param split_by_day: wether the results are computed first by day and averaged, or averaged globally
#         :return: mean of dictionary of metrics, std of dictionary of metrics
#         """
#         raw_results = self.compute_raw_results(split_by_day=split_by_day)
#
#         mean = {key: val for key, val in zip(list(raw_results.keys()), np.nanmean(list(raw_results.values()), axis=1))}
#         std = {key: val for key, val in zip(list(raw_results.keys()), np.nanstd(list(raw_results.values()), axis=1))}
#
#         return mean, std
#
#
# class ResultsDatasetTwoStepSmoothed():
#     def __init__(self, model, experiment, ph, dataset):
#         self.model = model
#         self.experiment = experiment
#         self.ph = ph
#         self.dataset = dataset
#         self.subjects = misc.datasets.datasets[dataset]["subjects"]
#         self.freq = misc.datasets.datasets[dataset]["glucose_freq"]
#
#     def compute_results_iter(self, iter=0, window_size=1, ratio=1, details=False):
#         """
#         Loop through the subjects of the dataset, and compute the mean performances
#         :return: mean of metrics, std of metrics
#         """
#         res = []
#         for subject in self.subjects:
#             res_subject = ResultsSubjectTwoStepSmoothed(self.model, self.experiment, self.ph, self.dataset,
#                                                 subject).compute_results_iter_smooth(iter, window_size,ratio)
#             if details:
#                 print(self.dataset, subject, res_subject)
#
#             res.append(res_subject[0])  # only the mean
#
#         keys = list(res[0].keys())
#         res = [list(res_.values()) for res_ in res]
#         mean, std = np.nanmean(res, axis=0), np.nanstd(res, axis=0)
#         return dict(zip(keys, mean)), dict(zip(keys, std))
#
#     def compute_results_all_iter(self,maxiter=30,window_size=1, ratio=1):
#         res = []
#         for iter in range(maxiter):
#             res.append(self.compute_results_iter(iter,window_size, ratio)[0])
#
#         return res
#
#     def save_all_iter_smooth(self, window_size, ratio):
#         dir = os.path.join(cs.path, "results", self.model, self.experiment, "smooth")
#         # file = os.path.join(dir, self.dataset + "_win"+str(window_size) + "_ratio"+str(ratio) + ".csv")
#         file = os.path.join(dir, self.dataset + "_es-ratio"+str(ratio) + ".csv")
#         Path(dir).mkdir(parents=True, exist_ok=True)
#         res = self.compute_results_all_iter(window_size=window_size, ratio=ratio)
#         data = [list(res_iter.values()) for res_iter in res]
#         df = pd.DataFrame(data=data,columns=list(res[0].keys()))
#         df.to_csv(file)