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


class ResultsSubjectAllIter(ResultsSubject):
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
        dict_arr = []
        for iter in range(len(self.results[0])):
            dict_arr.append(self.compute_results_iter_split(iter,split)[0])

        evolution = {}
        for key in dict_arr[0].keys():
            evolution[key] = [d[key] for d in dict_arr]

        return pd.DataFrame(data=np.transpose(list(evolution.values())),columns=list(evolution.keys()))
        # return np.transpose(list(evolution.values())), list(evolution.keys())

    def evolution_to_csv(self, split=0):
        evolution = self.compute_evolution(split)
        evolution.columns = [_.replace("_","-") for _ in evolution.columns]
        ega_cols = [col for col in evolution.columns if "EGA" in col]
        evolution.loc[:, ega_cols] = evolution.loc[:, ega_cols] * 100
        evolution.to_csv(os.path.join(cs.path,"tmp", "figures_data", "evolution_" + self.dataset + self.subject + ".dat"), index_label="iteration", sep=" ")

    def are_criteria_met(self, criteria_list):
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


class ResultsDatasetAllIter():
    def __init__(self, model, experiment, ph, dataset):
        self.model = model
        # self.experiment_valid = experiment + "_valid"
        self.experiment_test = experiment
        self.ph = ph
        self.dataset = dataset
        self.subjects = misc.datasets.datasets[dataset]["subjects"]
        self.freq = misc.datasets.datasets[dataset]["glucose_freq"]

    # def compute_results_with_criteria(self, criteria, threshold):
    #     if criteria == "AP":
    #         func = lambda res: np.where(np.array(res["CG_EGA_AP"]) >= threshold)[0]
    #     elif criteria == "BE":
    #         func = lambda res: np.where(np.array(res["CG_EGA_BE"]) < threshold)[0]
    #     elif criteria == "EP":
    #         func = lambda res: np.where(np.array(res["CG_EGA_EP"]) < threshold)[0]
    #     elif criteria == "MASE":
    #         func = lambda res: np.flip(np.where(np.array(res["MASE"]) < threshold))[0]
    #
    #     res = []
    #     for subject in self.subjects:
    #         evolution_valid = ResultsSubjectAllIter(self.model, self.experiment_valid, self.ph, self.dataset,
    #                                                 subject).compute_evolution()
    #         evolution_test = ResultsSubjectAllIter(self.model, self.experiment_test, self.ph, self.dataset,
    #                                                subject).compute_evolution()
    #         criteria_met_index = func(evolution_valid)
    #         if len(criteria_met_index) > 0:
    #             best_compromise_index = criteria_met_index[0]
    #             res.append({k: v[best_compromise_index] for k, v in evolution_test.items()})
    #         else:
    #             # criteria not met
    #             res.append({k: -1 for k, v in evolution_test.items()})
    #
    #     return res

    def compute_results_iter(self, iter=0, details=False):
        """
        Loop through the subjects of the dataset, and compute the mean performances
        :return: mean of metrics, std of metrics
        """
        res = []
        for subject in self.subjects:
            res_subject = ResultsSubjectAllIter(self.model, self.experiment_test, self.ph, self.dataset,
                                                subject).compute_results_iter_split(iter)
            if details:
                print(self.dataset, subject, res_subject)

            res.append(res_subject[0])  # only the mean

        keys = list(res[0].keys())
        res = [list(res_.values()) for res_ in res]
        mean, std = np.nanmean(res, axis=0), np.nanstd(res, axis=0)
        return dict(zip(keys, mean)), dict(zip(keys, std))

    def evolution_to_csv(self, split=0):
        for subject in misc.datasets.datasets[self.dataset]["subjects"]:
            ResultsSubjectAllIter(self.model, self.experiment_test, self.ph, self.dataset,
                                                subject).evolution_to_csv(split)

    def compute_results_all_iter(self,maxiter=30):
        res = []
        for iter in range(maxiter):
            res.append(self.compute_results_iter(iter)[0])

        return res

    def number_of_patients_meeting_criteria(self, criteria_list):
        """
        How to use : ResultsDatasetAllIter("pclstm","pclstm_iterative_gcmse_alliter",30,"idiab").number_of_patients_meeting_criteria([{"metric":"CG_EGA_AP","func": lambda x : np.greater_equal(x,0.90)}])

        :param criteria_list:
        :return:
        """
        count = 0
        for subject in misc.datasets.datasets[self.dataset]["subjects"]:
            if ResultsSubjectAllIter(self.model, self.experiment_test, self.ph, self.dataset,
                                                subject).are_criteria_met(criteria_list):
                print(subject + " meets criteria")
                count += 1

        return count

    def plot_evolution_through_iter(self, metric, save_file=None):
        res = []
        for subject in self.subjects:
            res.append(ResultsSubjectAllIter(self.model, self.experiment_test, self.ph, self.dataset,
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
            # np.savetxt(os.path.join(cs.path, "tmp", "figures_data",save_file + "_mean"),mean,header=np.r_[["iter"],keys])
            # np.savetxt(os.path.join(cs.path, "tmp", "figures_data",save_file + "_std"),std,header=np.r_[["iter"],keys])

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



    # def get_params_results_from_eval_mode(self, eval_mode):
    #     if eval_mode == "valid":
    #         return self.params_valid, self.results_valid
    #     elif eval_mode == "test":
    #         return self.params_test, self.results_test
    #     else:
    #         return self.params, self.results

    # def get_results_MASE_1(self):
    #     evolution


# def plot_steps_mase_region_ap(dataset, subject, exp, split=0):
#     import matplotlib.pyplot as plt
#     from postprocessing.metrics.mase import MASE
#     from postprocessing.metrics.cg_ega import CG_EGA
#     import numpy as np
#     from postprocessing.results import ResultsSubjectTwoStep
#     res = ResultsSubjectTwoStep("pclstm", exp, 30, dataset, subject).results
#     res = res[split]
#     freq = 15 if dataset == "idiab" else 5
#     rmse = [MASE(res_, 30, freq) for res_ in res]
#     cgega_region = np.array([CG_EGA(res_, freq).simplified() for res_ in res])
#     cgega_all = np.array([CG_EGA(res_, freq).reduced() for res_ in res])
#     iter = np.arange(len(rmse))
#     fig, ax1 = plt.subplots()
#     ax1.plot(iter, rmse, label="MASE")
#     ax1.set_xlabel("iter")
#     ax1.set_ylabel("MASE")
#     ax2 = ax1.twinx()
#     ax2.plot(iter, cgega_region[:, 0], label="AP-hypo", color="tab:green")
#     ax2.plot(iter, cgega_region[:, 3], label="AP-eu", color="tab:orange")
#     ax2.plot(iter, cgega_region[:, 6], label="AP-hyper", color="tab:red")
#     ax2.plot(iter, cgega_all[:, 0], label="AP", color="tab:purple")
#     plt.legend()
#     ax2.set_ylabel("probability")

# def hypo_mse_ratio(dataset, subject):
#     res = ResultsSubject("pclstm", "pclstm_mse_valid", 30, dataset, subject).results
#     ratio = []
#     for res_ in res:
#         res_hypo = res_.loc[res_.y_true < 70]
#         res_hyper = res_.loc[res_.y_true > 180]
#         mse_hypo = ((res_hypo.y_true - res_hypo.y_pred) ** 2).sum()
#         mse_hyper = ((res_hyper.y_true - res_hyper.y_pred) ** 2).sum()
#         ratio.append(mse_hypo/mse_hyper)
#         print(mse_hypo, mse_hyper)
#     return np.nanmean(ratio)

# a = []
# for sbj in ["559", "563", "570", "575", "588", "591"]:
#     a.append(hypo_mse_ratio("ohio", sbj))
#
# for sbj in ["1","2","3","4","5","6"]:
#     a.append(hypo_mse_ratio("idiab",sbj))


# def noap_pr_ratio(dataset, subject):
#     from postprocessing.metrics.cg_ega import CG_EGA
#     import misc
#     from postprocessing.results import ResultsSubject
#     res = ResultsSubject("pclstm", "pclstm_mse_valid", 30, dataset, subject).results
#     ratio = []
#     for res_ in res:
#         cgega = CG_EGA(res_,misc.datasets.datasets[dataset]["glucose_freq"]).per_sample()
#         # pega_cde = cgega[(cgega.P_EGA == "C") | (cgega.P_EGA == "D") | (cgega.P_EGA == "E")]
#         pega_cde = cgega
#         rega_cde = cgega[(cgega.R_EGA == "uC") | (cgega.R_EGA == "lC") | (cgega.R_EGA == "uD") | (cgega.R_EGA == "lD") | (cgega.R_EGA == "uE") | (cgega.R_EGA == "lE")]
#         pega_mse = ((pega_cde.y_true - pega_cde.y_pred) ** 2).mean()
#         rega_mse = ((rega_cde.dy_true - rega_cde.dy_pred) ** 2).mean()
#         ratio.append(rega_mse/pega_mse)
#         print(rega_mse, pega_mse)
#     return np.nanmean(ratio)
