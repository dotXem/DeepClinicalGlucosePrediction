import misc.constants as cs
import os
from misc.constants import path
from misc.utils import print_latex
import pandas as pd
from postprocessing.results import ResultsSubject
import matplotlib.pyplot as plt
from scipy.stats import kde
import numpy as np
import misc.datasets
from postprocessing.metrics.cg_ega import CG_EGA
from postprocessing.metrics.p_ega import P_EGA
from postprocessing.metrics.r_ega import R_EGA


class ResultsAnalyzer:
    def __init__(self, model, exp):
        self.model = model
        self.exp = exp
        pass

    def plot_mse_contrib(self, dataset, subject, nbins=25):
        res = self._get_res(self.model, self.exp, dataset, subject)

        plt.figure()
        plt.hist2d(res.y_true.values, res.y_pred.values, range=[[0, 400], [0, 400]], bins=nbins, cmap="Blues",
                   density=True, weights=res.error)
        cb = plt.colorbar()
        cb.set_label('weighed_counts in bin')
        self._plot_pega_grid()

    def plot_density_errors(self, dataset, weights=False):
        fig, axes = plt.subplots(ncols=len(misc.datasets.datasets[dataset]["subjects"]), nrows=1, figsize=(21, 5))
        for i, sbj, in enumerate(misc.datasets.datasets[dataset]["subjects"]):
            self.plot_density_errors_per_subject(dataset, sbj, ax=axes[i], weights=weights)
            axes[i].set(aspect='equal')

    def plot_error_distributions(self, dataset, nbins=50):
        n_arr, bins_arr = [], []
        for i, subject, in enumerate(misc.datasets.datasets[dataset]["subjects"]):
            res = self._get_res(self.model, self.exp, dataset, subject)
            (n, bins, _) = plt.hist(res.error, bins=nbins, range=[0, 400], density=True, stacked=True)
            n_arr.append(n)
            bins_arr.append(bins)
        plt.close()

        bins_arr = np.array(bins_arr)
        n_arr = np.array(n_arr) * 400 / nbins
        middle_bins = ((bins_arr[:, 1:] + bins_arr[:, :-1]) / 2)[0]
        mean = np.mean(n_arr, axis=0)
        std = np.std(n_arr, axis=0)
        plt.figure()
        plt.plot(middle_bins, mean, color='#CC4F1B')

        plt.fill_between(middle_bins, mean - std, mean + std,
                         alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        plt.title("Distribution des erreurs de prédiction de glycémie pour le jeu de données " + dataset + ".")
        plt.xlabel("erreur de glycémie [mg/dL]")
        plt.ylabel("probabilité")

    def plot_error_vs_glucose_true(self, dataset):
        for i, subject, in enumerate(misc.datasets.datasets[dataset]["subjects"]):
            res = self._get_res(self.model, self.exp, dataset, subject)
            plt.scatter(res.y_true, res.error)

    def mse_impact_per_pega_zone(self, dataset):

        mse_impact_per_zone = []
        for i, sbj, in enumerate(misc.datasets.datasets[dataset]["subjects"]):
            mse_impact_per_zone.append(self.mse_impact_per_pega_zone_per_subject(dataset, sbj))

        mse_impact_per_zone = pd.DataFrame(data=mse_impact_per_zone, columns=["A", "B", "C", "D", "E"],
                                           index=misc.datasets.datasets[dataset]["subjects"])
        mean = mse_impact_per_zone.mean(axis=0)
        std = mse_impact_per_zone.std(axis=0)
        mse_impact_per_zone.loc["mean"] = mean
        mse_impact_per_zone.loc["std"] = std

        return mse_impact_per_zone

    def mse_impact_per_pega_zone_per_subject(self, dataset, subject):
        res = ResultsSubject(self.model, self.exp, 30, dataset, subject).results
        cg_ega_arr = [CG_EGA(res_split, misc.datasets.datasets[dataset]["glucose_freq"]).per_sample().dropna() for
                      res_split in
                      res]

        a_impact, b_impact, c_impact, d_impact, e_impact = [], [], [], [], []
        for cg_ega_split in cg_ega_arr:
            sum = ((cg_ega_split.y_true - cg_ega_split.y_pred) ** 2).sum()
            for impact_arr, zone in zip([a_impact, b_impact, c_impact, d_impact, e_impact], ["A", "B", "C", "D", "E"]):
                subset = cg_ega_split.loc[cg_ega_split.P_EGA == zone]
                impact_arr.append(((subset.y_true - subset.y_pred) ** 2).sum() / sum)

            print(np.sum([a_impact[-1], b_impact[-1], c_impact[-1], d_impact[-1], e_impact[-1]]))

        a_impact, b_impact, c_impact, d_impact, e_impact = [np.nanmean(impact) for impact in
                                                            [a_impact, b_impact, c_impact, d_impact, e_impact]]

        # sums = [ for cg_ega_split in cg_ega_arr]

        # def proportion_calc(impact, sums):
        #     impact = np.mean([impact_split / sums_split for impact_split, sums_split in zip(impact, sums)])
        #     return impact
        #
        # a_impact, b_impact, c_impact, d_impact, e_impact = [proportion_calc(impact, sums) for impact in
        #                                                     [a_impact, b_impact, c_impact, d_impact, e_impact]]

        return a_impact, b_impact, c_impact, d_impact, e_impact

    def plot_glucose_distributions(self, dataset, nbins=50, export=False):
        n_arr, bins_arr = [], []
        for i, subject, in enumerate(misc.datasets.datasets[dataset]["subjects"]):
            res = self._get_res(self.model, self.exp, dataset, subject)
            (n, bins, _) = plt.hist(res.y_true, bins=nbins, range=[0, 400], density=True, stacked=True)
            n_arr.append(n)
            bins_arr.append(bins)
        plt.close()

        bins_arr = np.array(bins_arr)
        n_arr = np.array(n_arr) * 400 / nbins
        middle_bins = ((bins_arr[:, 1:] + bins_arr[:, :-1]) / 2)[0]
        mean = np.mean(n_arr, axis=0)
        std = np.std(n_arr, axis=0)
        plt.figure()
        plt.plot(middle_bins, mean, color='#CC4F1B')

        plt.fill_between(middle_bins, mean - std, mean + std,
                         alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        plt.title("Distribution des échantillons de glycémie pour le jeu de données " + dataset + ".")
        plt.xlabel("glycémie [mg/dL]")
        plt.ylabel("probabilité")

        if export:
            df =pd.DataFrame(data=np.c_[middle_bins, mean,mean-std,mean+std], columns=["middle_bins","mean","plus-std","minus-std"])
            df.to_csv(path.join(cs.path, "tmp","figures_data","glucose_distribution_"+dataset+".dat"),index_label="index")

    def plot_density_errors_per_subject(self, dataset, subject, nbins=25, ax=None, weights=False):
        res = self._get_res(self.model, self.exp, dataset, subject)
        weights = res.error if weights else None
        k = kde.gaussian_kde([res.y_true, res.y_pred], weights=weights)
        xi, yi = np.mgrid[0:400:nbins * 1j, 0:400:nbins * 1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        if ax is None:
            _, ax = plt.subplots(1, 1)

        ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap="Purples")  # cmap=plt.cm.BuGn_r)
        self._plot_pega_grid(ax)

    def erroneous_prediction_analysis(self, dataset, disp=False):
        cg_ega_analysis = []
        for i, sbj, in enumerate(misc.datasets.datasets[dataset]["subjects"]):
            mean_sbj, std_sbj = self.erroneous_prediction_analysis_per_subject(dataset, sbj)
            cg_ega_analysis.append(mean_sbj)

            if disp:
                print_latex(mean_sbj, std_sbj, label=sbj)

        cg_ega_analysis = pd.DataFrame(data=cg_ega_analysis, columns=["Hypo_EP_P", "Hypo_EP_R", "Hypo_EP_B",
                                                                      "Eu_EP_P", "Eu_EP_R", "Eu_EP_B",
                                                                      "Hyper_EP_P", "Hyper_EP_R", "Hyper_EP_B"
                                                                      ],
                                       index=misc.datasets.datasets[dataset]["subjects"])
        mean = cg_ega_analysis.mean(axis=0)
        std = cg_ega_analysis.std(axis=0)
        cg_ega_analysis.loc["mean"] = mean
        cg_ega_analysis.loc["std"] = std

        if disp:
            print_latex(mean, std, label=dataset)

        return cg_ega_analysis

    def plot_importance_bad_pega_in_loss(self,c=None):

        scaling_arr = np.logspace(0, 10, num=21, base=2)

        cde_impact_ds, r_impact_ds = [], []
        for dataset in ["idiab","ohio"]:
            cde_impact_sbj, r_impact_sbj = [], []
            for subject in misc.datasets.datasets[dataset]["subjects"]:
                res = ResultsSubject(self.model, self.exp, 30, dataset, subject).results
                cg_ega_arr = [CG_EGA(res_split, misc.datasets.datasets[dataset]["glucose_freq"]).per_sample()
                              for res_split in res]

                cde_impact_arr, r_impact_arr = [], []
                for scaling in scaling_arr:
                    cde_impact_tmp, r_impact_tmp = [], []
                    for cg_ega_split in cg_ega_arr:
                        ab_ind = (cg_ega_split["P_EGA"] == "A") | (cg_ega_split["P_EGA"] == "B")
                        cg_ega_ab, cg_ega_cde = cg_ega_split.loc[ab_ind], cg_ega_split.loc[~ab_ind]
                        ab_loss = ((cg_ega_ab.y_true - cg_ega_ab.y_pred) ** 2).sum() / scaling
                        cde_loss = ((cg_ega_cde.y_true - cg_ega_cde.y_pred) ** 2).sum()
                        if c is not None:
                            r_ega = cg_ega_split.loc[~((cg_ega_split["R_EGA"] == "A") | (cg_ega_split["R_EGA"] == "B"))]
                            r_loss = ((r_ega.dy_true - r_ega.dy_pred) ** 2).sum() * c
                            total_loss = ab_loss + cde_loss + r_loss
                            r_impact_in_loss = r_loss / total_loss
                            r_impact_tmp.append(r_impact_in_loss)
                        else:
                            total_loss = ab_loss + cde_loss
                        cde_impact_in_loss = cde_loss / total_loss
                        cde_impact_tmp.append(cde_impact_in_loss)

                    cde_impact_arr.append(np.mean(cde_impact_tmp))
                    if c is not None:
                        r_impact_arr.append(np.mean(r_impact_tmp))

                cde_impact_sbj.append(cde_impact_arr)
                if c is not None:
                    r_impact_sbj.append(r_impact_arr)

            cde_impact_ds.append([np.mean(cde_impact_sbj,axis=0), np.std(cde_impact_sbj,axis=0)])
            if c is not None:
                r_impact_ds.append([np.mean(r_impact_sbj,axis=0), np.std(r_impact_sbj,axis=0)])

        for ds, edgecolor, facecolor in zip(cde_impact_ds, ["#25416d", "#004e11"], ["#afccff", "#b7ffc5"]):
            plt.fill_between(scaling_arr, ds[0] - ds[1], ds[0] + ds[1], alpha=0.5, edgecolor=edgecolor, facecolor=facecolor)

        if c is not None:
            for ds, edgecolor, facecolor in zip(r_impact_ds, ["#8e0606",'#CC4F1B'], ["#ffb7b7",'#FF9848']):
                plt.fill_between(scaling_arr, ds[0] - ds[1], ds[0] + ds[1], alpha=0.5, edgecolor=edgecolor, facecolor=facecolor)

        if c is not None:
            plt.legend(["idiab_p_cde_impact", "ohio_p_cde_impact", "idiab_r_impact", "ohio_r_impact"])
        else:
            plt.legend(["idiab_p_cde_impact", "ohio_p_cde_impact"])

        for ds, edgecolor, facecolor in zip(cde_impact_ds, ["#25416d", "#004e11"], ["#afccff", "#b7ffc5"]):
            plt.plot(scaling_arr, ds[0], color=edgecolor)

        if c is not None:
            for ds, edgecolor, facecolor in zip(r_impact_ds, ["#8e0606",'#CC4F1B'], ["#ffb7b7",'#FF9848']):
                plt.plot(scaling_arr, ds[0], color=edgecolor)

        plt.xlabel("P-A/B scaling factor")
        plt.ylabel("relative importance of P-C/D/E samples on loss")
        plt.title("relative importance of P-C/D/E (and R) on loss depending on P-A/B scaling factor")

    def p_ega_analysis(self, dataset):
        # fig, axes = plt.subplots(ncols=len(misc.datasets.datasets[dataset]["subjects"]), nrows=1, figsize=(21, 5))
        p_ega_analysis = []
        for i, sbj, in enumerate(misc.datasets.datasets[dataset]["subjects"]):
            p_ega_analysis.append(self.p_ega_analysis_per_subject(dataset, sbj))
        p_ega_analysis = pd.DataFrame(data=p_ega_analysis, columns=["A+B", "A", "B", "C", "D", "E"],
                                      index=misc.datasets.datasets[dataset]["subjects"])
        mean = p_ega_analysis.mean(axis=0)
        std = p_ega_analysis.std(axis=0)
        p_ega_analysis.loc["mean"] = mean
        p_ega_analysis.loc["std"] = std

        return p_ega_analysis

    def r_ega_analysis(self, dataset):
        # fig, axes = plt.subplots(ncols=len(misc.datasets.datasets[dataset]["subjects"]), nrows=1, figsize=(21, 5))
        r_ega_analysis = []
        for i, sbj, in enumerate(misc.datasets.datasets[dataset]["subjects"]):
            r_ega_analysis.append(self.r_ega_analysis_per_subject(dataset, sbj))
        r_ega_analysis = pd.DataFrame(data=r_ega_analysis,
                                      columns=["A+B", "A", "B", "uC", "lC", "uD", "lD", "uE", "lE"],
                                      index=misc.datasets.datasets[dataset]["subjects"])
        mean = r_ega_analysis.mean(axis=0)
        std = r_ega_analysis.std(axis=0)
        r_ega_analysis.loc["mean"] = mean
        r_ega_analysis.loc["std"] = std

        return r_ega_analysis

    def p_ega_analysis_per_subject(self, dataset, subject):
        res = ResultsSubject(self.model, self.exp, 30, dataset, subject).results
        p_ega_arr = np.array([P_EGA(res_split, misc.datasets.datasets[dataset]["glucose_freq"]).mean() for res_split in
                              res])

        p_ega_arr = np.c_[np.sum(p_ega_arr[:, :2], axis=1).reshape(-1, 1), p_ega_arr]
        p_ega_arr = np.nanmean(p_ega_arr, axis=0)

        return p_ega_arr

    def r_ega_analysis_per_subject(self, dataset, subject):
        res = ResultsSubject(self.model, self.exp, 30, dataset, subject).results
        r_ega_arr = np.array([R_EGA(res_split, misc.datasets.datasets[dataset]["glucose_freq"]).mean() for res_split in
                              res])

        r_ega_arr = np.c_[np.sum(r_ega_arr[:, :2], axis=1).reshape(-1, 1), r_ega_arr]
        r_ega_arr = np.nanmean(r_ega_arr, axis=0)

        return r_ega_arr

    def erroneous_prediction_analysis_per_subject(self, dataset, subject):
        res = ResultsSubject(self.model, self.exp, 30, dataset, subject).results
        cg_ega_arr = [CG_EGA(res_split, misc.datasets.datasets[dataset]["glucose_freq"]).per_sample() for res_split in
                      res]

        hypo, eu, hyper = [], [], []
        for cg_ega_split in cg_ega_arr:
            cg_ega_split["reason"] = ""
            cg_ega_split = cg_ega_split.loc[cg_ega_split.CG_EGA == "EP"].dropna()
            hypo_split = cg_ega_split[cg_ega_split.y_true <= 70]
            eu_split = cg_ega_split[(cg_ega_split.y_true > 70) & (cg_ega_split.y_true <= 180)]
            hyper_split = cg_ega_split[cg_ega_split.y_true > 180]

            if not hypo_split.empty:
                hypo_split.loc[(hypo_split.P_EGA == "A"), "reason"] = "R"
                hypo_split.loc[(~(hypo_split.P_EGA == "A")), "reason"] = "B"
                hypo_split.loc[((hypo_split.R_EGA == "A") | (hypo_split.R_EGA == "B")), "reason"] = "P"
            hypo.append(hypo_split)

            if not eu_split.empty:
                eu_split.loc[((eu_split.P_EGA == "A") | (eu_split.P_EGA == "B")), "reason"] = "R"
                eu_split.loc[(eu_split.P_EGA == "C"), "reason"] = "B"
                eu_split.loc[(eu_split.P_EGA == "C") & (
                        (eu_split.R_EGA == "A") | (eu_split.R_EGA == "B")), "reason"] = "P"
            eu.append(eu_split)

            if not hyper_split.empty:
                hyper_split.loc[((hyper_split.P_EGA == "A") | (hyper_split.P_EGA == "B")), "reason"] = "R"
                hyper_split.loc[(hyper_split.P_EGA == "C") | (hyper_split.P_EGA == "D") | (
                            hyper_split.P_EGA == "E"), "reason"] = "B"
                hyper_split.loc[((hyper_split.R_EGA == "A") | (hyper_split.R_EGA == "B")) & (
                        (hyper_split.P_EGA == "C") | (hyper_split.P_EGA == "D") | (
                        hyper_split.P_EGA == "E")), "reason"] = "P"
            hyper.append(hyper_split)

        def stats_region_reason(region):
            total_arr = np.array([split.__len__() for split in region])
            p_arr = np.array([split.loc[(split.reason == "P")].__len__() for split in region]) / total_arr
            r_arr = np.array([split.loc[(split.reason == "R")].__len__() for split in region]) / total_arr
            b_arr = np.array([split.loc[(split.reason == "B")].__len__() for split in region]) / total_arr
            return [np.nanmean(p_arr), np.nanmean(r_arr), np.nanmean(b_arr)], [np.nanstd(p_arr), np.nanstd(r_arr),
                                                                               np.nanstd(b_arr)]

        hypo_ep_mean, hypo_ep_std = stats_region_reason(hypo)
        eu_ep_mean, eu_ep_std = stats_region_reason(eu)
        hyper_ep_mean, hyper_ep_std = stats_region_reason(hyper)

        return np.r_[hypo_ep_mean, eu_ep_mean, hyper_ep_mean], np.r_[hypo_ep_std, eu_ep_std, hyper_ep_std]

    def _get_res(self, model, exp, dataset, subject):
        # TODO res.results[0] ??? need args "split" ?
        res = ResultsSubject(model, exp, 30, dataset, subject)
        res_raw = res.results[0]
        res_raw.loc[:, "error"] = (res_raw.y_true - res_raw.y_pred) ** 2
        res_raw = res_raw.dropna()
        return res_raw

    def _plot_pega_grid(self, ax):
        ax.plot([0, 400], [0, 400], "-k")
        ax.plot([58.33, 400], [58.33333 * 6 / 5, 400 * 6 / 5], "-k")
        ax.plot([0, 58.33333], [70, 70], "-k")
        ax.plot([70, 400], [56, 320], "-k")
        ax.plot([70, 70], [0, 56], "-k")
        ax.plot([70, 70], [84, 400], "-k")
        ax.plot([0, 70], [180, 180], "-k")
        ax.plot([70, 400], [70 * 22 / 17 + 89.412, 400 * 22 / 17 + 89.412], "-k")
        ax.plot([180, 180], [0, 70], "-k")
        ax.plot([180, 400], [70, 70], "-k")
        ax.plot([240, 240], [70, 180], "-k")
        ax.plot([240, 400], [180, 180], "-k")
        ax.plot([130, 180], [130 * 7 / 5 - 182, 180 * 7 / 5 - 182], "-k")
        ax.plot([130, 180], [130 * 7 / 5 - 202, 180 * 7 / 5 - 202], "--k")
        ax.plot([180, 400], [50, 50], "--k")
        ax.plot([240, 400], [160, 160], "--k")
        ax.plot([58.33333, 400], [58.33333 * 6 / 5 + 20, 400 * 6 / 5 + 20], "--k")
        ax.plot([0, 58.33333], [90, 90], "--k")
        ax.plot([0, 70], [200, 200], "--k")
        ax.plot([70, 400], [70 * 22 / 17 + 109.412, 400 * 22 / 17 + 109.412], "--k")
        ax.text(38, 12, "A")
        ax.text(12, 38, "A")
        ax.text(375, 240, "B")
        ax.text(260, 375, "B")
        ax.text(150, 375, "C")
        ax.text(165, 25, "C")
        ax.text(25, 125, "D")
        ax.text(375, 125, "D")
        ax.text(375, 25, "E")
        ax.text(25, 375, "E")
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 400)

    def save_all_p_r_ega_points(self, dataset):
        ap_p, be_p, ep_p, ap_r, be_r, ep_r = [], [], [], [], [], []
        for i, sbj, in enumerate(misc.datasets.datasets[dataset]["subjects"]):
            for split_res in ResultsSubject(self.model, self.exp, 30, dataset, sbj).results:
                cg_ega = CG_EGA(split_res, misc.datasets.datasets[dataset]["glucose_freq"]).per_sample()
                ap_p.append(cg_ega.loc[cg_ega.CG_EGA == "AP", ["y_true", "y_pred"]].values)
                be_p.append(cg_ega.loc[cg_ega.CG_EGA == "BE", ["y_true", "y_pred"]].values)
                ep_p.append(cg_ega.loc[cg_ega.CG_EGA == "EP", ["y_true", "y_pred"]].values)
                ap_r.append(cg_ega.loc[cg_ega.CG_EGA == "AP", ["dy_true", "dy_pred"]].values)
                be_r.append(cg_ega.loc[cg_ega.CG_EGA == "BE", ["dy_true", "dy_pred"]].values)
                ep_r.append(cg_ega.loc[cg_ega.CG_EGA == "EP", ["dy_true", "dy_pred"]].values)

        ap_p = np.concatenate(ap_p, axis=0).reshape(-1, 2)
        be_p = np.concatenate(be_p, axis=0).reshape(-1, 2)
        ep_p = np.concatenate(ep_p, axis=0).reshape(-1, 2)
        ap_r = np.concatenate(ap_r, axis=0).reshape(-1, 2)
        be_r = np.concatenate(be_r, axis=0).reshape(-1, 2)
        ep_r = np.concatenate(ep_r, axis=0).reshape(-1, 2)

        np.savetxt(os.path.join(path, "tmp", "figures_data", "P-EGA_AP_" + dataset + ".dat"), ap_p)
        np.savetxt(os.path.join(path, "tmp", "figures_data", "P-EGA_BE_" + dataset + ".dat"), be_p)
        np.savetxt(os.path.join(path, "tmp", "figures_data", "P-EGA_EP_" + dataset + ".dat"), ep_p)
        np.savetxt(os.path.join(path, "tmp", "figures_data", "R-EGA_AP_" + dataset + ".dat"), ap_r)
        np.savetxt(os.path.join(path, "tmp", "figures_data", "R-EGA_BE_" + dataset + ".dat"), be_r)
        np.savetxt(os.path.join(path, "tmp", "figures_data", "R-EGA_EP_" + dataset + ".dat"), ep_r)
