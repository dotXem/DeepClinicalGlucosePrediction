import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import misc.constants
import misc.datasets
from misc import constants as cs
from misc.utils import print_latex
from postprocessing.results import ResultsSubject


class ResultsSubjectPICA(ResultsSubject):
    """
    Same as ResultsSubject object from postprocessing.results, but works with all iterations of algorithm APAC at once

    Posess several additional functions:
        - compute_evolution
        - evolution_to_csv
        - are_criteria_met
    """
    def __init__(self, model, experiment, ph, dataset, subject, params=None, results=None):

        self.model = model
        self.experiment = experiment
        self.ph = ph
        self.dataset = dataset
        self.subject = subject
        self.freq = np.max([misc.constants.freq, misc.datasets.datasets[dataset]["glucose_freq"]])

        if results is None and params is None:
            self.params, self.results = self.load_raw_results()
        else:
            self.results = results
            self.params = params

    def load_raw_results(self, legacy=False):
        """
        Load the results from previous instance of ResultsSubject that has saved the them
        :param legacy: if legacy object shall  be used
        :return: (params dictionary), dataframe with ground truths and predictions
        """
        file = self.dataset + "_" + self.subject + ".npy"
        path = os.path.join(cs.path, "results", self.model, self.experiment, "ph-" + str(self.ph), file)

        params, results = np.load(path, allow_pickle=True)
        dfs = []
        for results_cv in results:
            dfs_cv = []
            for results_iter in results_cv:
                df = pd.DataFrame(results_iter, columns=["datetime", "y_true", "y_pred"])
                df = df.set_index("datetime")
                df = df.astype("float32")
                dfs_cv.append(df)
            dfs.append(dfs_cv)
        return params, dfs

    def save_raw_results(self):
        """
        Save the results and params
        :return:
        """
        dir = os.path.join(cs.path, "results", self.model, self.experiment, "ph-" + str(self.ph))
        Path(dir).mkdir(parents=True, exist_ok=True)

        saveable_results = np.array([[res_.reset_index().to_numpy() for res_ in res] for res in self.results])

        np.save(os.path.join(dir, self.dataset + "_" + self.subject + ".npy"), [self.params, saveable_results])

    def compute_results_iter_split(self, iter=0, split=0):
        res = [self.results[split][iter]]
        results_subject = ResultsSubject(self.model, self.experiment, self.ph, self.dataset, self.subject, self.params,
                                         res)
        return results_subject.compute_mean_std_results()

    def compute_results_all_iter(self, split=0):
        res_iter = []
        for iter in range(len(self.results[0])):
            res_iter.append(self.compute_results_iter_split(iter,split)[0])
        return res_iter

    def compute_evolution(self,split=0):
        """
        Compute the evolution of each metric through the algorithm APAC
        :param split: number of given split
        :return: DataFrame of shape (n_iter, n_metrics)
        """
        dict_arr = []
        for iter in range(len(self.results[0])):
            dict_arr.append(self.compute_results_iter_split(iter,split)[0])

        evolution = {}
        for key in dict_arr[0].keys():
            evolution[key] = [d[key] for d in dict_arr]

        return pd.DataFrame(data=np.transpose(list(evolution.values())),columns=list(evolution.keys()))

    def evolution_to_csv(self, split=0):
        """
        Save the evolution computed by function compute_evolution into csv file format
        :param split: number of given split
        :return:
        """
        evolution = self.compute_evolution(split)
        evolution.columns = [_.replace("_","-") for _ in evolution.columns]
        ega_cols = [col for col in evolution.columns if "EGA" in col]
        evolution.loc[:, ega_cols] = evolution.loc[:, ega_cols] * 100
        evolution.to_csv(os.path.join(cs.path,"tmp", "figures_data", "evolution_" + self.dataset + self.subject + ".dat"), index_label="iteration", sep=" ")

    def are_criteria_met(self, criteria_list):
        """
        From the evolution DataFrame computed by the compute_evolution function, return True if all the criterias in
        criteria_list parameter are met by the subject else False

        criteria example : {"metric":"CG_EGA_AP","func": lambda x : np.greater_equal(x,0.90)}
        :param criteria_list: list of criteria having the shape of dict with keys "metric" and "func" (see example above)
        :return: True if all the criterias in criteria_list parameter are met by the subject else False
        """
        evolution = self.compute_evolution()
        criteria_meeting = []
        for criteria in criteria_list:
            criteria_meeting.append(criteria["func"](np.array(evolution[criteria["metric"]])))

        return np.any(np.all(criteria_meeting,axis=0))

    def compute_mean_std_results(self, split_by_day=False):
        """
        From the raw metrics scores, compute the mean and std
        :param split_by_day: wether the results are computed first by day and averaged, or averaged globally
        :return: mean of dictionary of metrics, std of dictionary of metrics
        """
        raw_results = self.compute_raw_results(split_by_day=split_by_day)

        mean = {key: val for key, val in zip(list(raw_results.keys()), np.nanmean(list(raw_results.values()), axis=1))}
        std = {key: val for key, val in zip(list(raw_results.keys()), np.nanstd(list(raw_results.values()), axis=1))}

        return mean, std


class ResultsDatasetPICA():
    """
    Same as ResultsDataset object from postprocessing.results, but works with all iterations of algorithm APAC at once

    Posess several additional functions:
        - evolution_to_csv
        - number_of_subjects_meeting_criteria
        - plot_evolution_through_iter
    """
    def __init__(self, model, experiment, ph, dataset):
        self.model = model
        self.experiment_test = experiment
        self.ph = ph
        self.dataset = dataset
        self.subjects = misc.datasets.datasets[dataset]["subjects"]
        self.freq = misc.datasets.datasets[dataset]["glucose_freq"]

    def compute_results_iter(self, iter=0, details=False):
        """
        Loop through the subjects of the dataset, and compute the mean performances
        :return: mean of metrics, std of metrics
        """
        res = []
        for subject in self.subjects:
            res_subject = ResultsSubjectPICA(self.model, self.experiment_test, self.ph, self.dataset,
                                             subject).compute_results_iter_split(iter)
            if details:
                print(self.dataset, subject, res_subject)

            res.append(res_subject[0])  # only the mean

        keys = list(res[0].keys())
        res = [list(res_.values()) for res_ in res]
        mean, std = np.nanmean(res, axis=0), np.nanstd(res, axis=0)
        return dict(zip(keys, mean)), dict(zip(keys, std))

    def evolution_to_csv(self, split=0):
        """
        Save into CSV the evolution of all dataset's subjects
        :param split: number of given split
        :return:
        """
        for subject in misc.datasets.datasets[self.dataset]["subjects"]:
            ResultsSubjectPICA(self.model, self.experiment_test, self.ph, self.dataset,
                               subject).evolution_to_csv(split)

    def compute_results_all_iter(self,maxiter=30):
        res = []
        for iter in range(maxiter):
            res.append(self.compute_results_iter(iter)[0])

        return res

    def number_of_subjects_meeting_criteria(self, criteria_list):
        """
        How to use : ResultsDatasetAllIter("pclstm","pclstm_iterative_gcmse_alliter",30,"idiab")\
            .number_of_patients_meeting_criteria([{"metric":"CG_EGA_AP","func": lambda x : np.greater_equal(x,0.90)}])

        :param criteria_list: list of criteria like {"metric":"CG_EGA_AP","func": lambda x : np.greater_equal(x,0.90)}
        :return: count of subjects meeting criteria
        """
        count = 0
        for subject in misc.datasets.datasets[self.dataset]["subjects"]:
            if ResultsSubjectPICA(self.model, self.experiment_test, self.ph, self.dataset,
                                  subject).are_criteria_met(criteria_list):
                print(subject + " meets criteria")
                count += 1

        return count

    def plot_evolution_through_iter(self, metric, save_file=None):
        """ plot average evolution of given metric through APAC iteration"""
        res = []
        for subject in self.subjects:
            res.append(ResultsSubjectPICA(self.model, self.experiment_test, self.ph, self.dataset,
                                          subject).compute_results_all_iter())

        iter = np.arange(len(res[0]))
        keys = np.array(list(res[0][0].keys()))
        metric_index = np.where(keys == metric)[0][0]
        # vals = res
        vals = np.array(
            [[list(res[sbj_i][iter_i].values()) for iter_i in range(len(res[0]))] for sbj_i in range(len(res))])

        vals_mean = np.nanmean(vals, axis=0)
        vals_std = np.nanstd(vals, axis=0)

        mean,std = vals_mean[:,metric_index], vals_std[:,metric_index]

        plt.figure()
        plt.plot(iter, mean, color='#CC4F1B')

        plt.fill_between(iter, mean - std, mean + std,
                         alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        plt.title(metric + " en fonction de iter")
        plt.xlabel("iter")
        plt.ylabel(metric)

        from pandas import DataFrame

        if save_file is not None:
            path = os.path.join(cs.path, "tmp", "figures_data",save_file)
            DataFrame(data=np.c_[iter.reshape(-1,1), vals_mean],columns= np.r_[["iter"],keys]).to_csv(path + "_mean",sep=" ")
            DataFrame(data=np.c_[iter.reshape(-1,1), vals_std],columns= np.r_[["iter"],keys]).to_csv(path + "_std",sep=" ")

    def to_latex(self, iter=0, table="general", model_name=None):
        """
        Format the results into a string for the paper in LATEX
        :param table: either "acc" or "cg_ega", corresponds to the table
        :param model_name: prefix of the string, name of the model
        :return:
        """
        mean, std = self.compute_results_iter(iter)
        if table == "p_ega":
            p_ega_keys = ["P_EGA_A+B", "P_EGA_A", "P_EGA_B", "P_EGA_C", "P_EGA_D", "P_EGA_E"]
            mean = [mean[k] * 100 for k in p_ega_keys]
            std = [std[k] * 100 for k in p_ega_keys]

        elif table == "r_ega":
            r_ega_keys = ["R_EGA_A+B", "R_EGA_A", "R_EGA_B", "R_EGA_lC", "R_EGA_uC", "R_EGA_lD", "R_EGA_uD", "R_EGA_lE",
                          "R_EGA_uC"]
            mean = [mean[k] * 100 for k in r_ega_keys]
            std = [std[k] * 100 for k in r_ega_keys]
        elif table == "cg_ega":
            cg_ega_keys = ["CG_EGA_AP_hypo", "CG_EGA_BE_hypo", "CG_EGA_EP_hypo", "CG_EGA_AP_eu", "CG_EGA_BE_eu",
                           "CG_EGA_EP_eu", "CG_EGA_AP_hyper", "CG_EGA_BE_hyper", "CG_EGA_EP_hyper"]
            mean = [mean[k] * 100 for k in cg_ega_keys]
            std = [std[k] * 100 for k in cg_ega_keys]
        elif table == "acc":
            acc_keys = ["RMSE", "MAPE", "MASE", "TG"]
            mean = [mean[k] for k in acc_keys]
            std = [std[k] for k in acc_keys]
        elif table == "general":
            acc_keys = ["RMSE", "MAPE", "MASE", "TG", "CG_EGA_AP", "CG_EGA_BE", "CG_EGA_EP"]
            mean = [mean[k] if k not in ["CG_EGA_AP", "CG_EGA_BE", "CG_EGA_EP"] else mean[k] * 100 for k in acc_keys]
            std = [std[k] if k not in ["CG_EGA_AP", "CG_EGA_BE", "CG_EGA_EP"] else std[k] * 100 for k in acc_keys]

        print_latex(mean, std, label=self.model)


