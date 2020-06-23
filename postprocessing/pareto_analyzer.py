from misc.utils import printd
import matplotlib.pyplot as plt
import sys
import pandas as pd
import os
import numpy as np
import re


class ParetoAnalyzer():
    max_metrics = ["TG", "CG_EGA_AP", "CG_EGA_AP_hypo", "CG_EGA_AP_eu", "CG_EGA_AP_hyper", "P_EGA_A+B", "R_EGA_A+B"]

    def __init__(self, model, dataset, ph, experiments):
        """

        :param model: model name (e.g., "pclstm")
        :param dataset: dataset name (e.g., "idiab")
        :param ph: prediction horizon (e.g., 30)
        :param experiments: list of experiment names to load from
        """
        self.model = model
        self.dataset = dataset
        self.ph = ph
        self.experiments = experiments
        self.results = self._load_experiments(experiments)

    def _load_experiments(self, experiments):
        df_experiments = []
        for experiment in experiments:
            c = float(re.search("_c(.*)", experiment).group(1))
            dir = os.path.join("results", self.model, experiment, "smooth")
            for file in os.listdir(dir):
                if self.dataset in file:
                    df = pd.read_csv(os.path.join(dir, file))
                    [win, ratio] = re.findall(r"[-+]?\d*\.\d+|\d+", file)
                    df["c"] = np.full((len(df.index)), c)
                    df["iter"] = df.index
                    df["p_ab"] = df.iter
                    df.loc[df.p_ab == 0, "p_ab"] = 1
                    df.p_ab = 1 / df.p_ab
                    df["win"] = np.full((len(df.index)), int(win))
                    df["ratio"] = np.full((len(df.index)), float(ratio))
                    df_experiments.append(df)
        return pd.concat(df_experiments).reset_index()

    def plot_solutions(self, metrics, highlight=None):
        """
        plot all solutions given the objective (metrics)
        :param metrics: list of metrics (e.g., ["CG_EGA_AP","RMSE"]), min 2
        :param highlight: dict with keys "c", "win", "ratio"; used to highlight specific results
        :return:
        """
        if len(metrics) < 2:
            sys.exit()

        pareto_front = self.undominated_solutions(metrics)

        for i in range(len(metrics) - 1):
            for j in range(i + 1, len(metrics)):
                pareto_front = pareto_front.sort_values(
                    metrics[i])  # potentiellement, toutes les solutions sont plot quand beaucoup de metrics
                plt.figure()
                plt.plot(self.results.loc[:, metrics[i]], self.results.loc[:, metrics[j]], "o")
                plt.plot(pareto_front.loc[:, metrics[i]], pareto_front.loc[:, metrics[j]], "k-")

                if highlight is not None:
                    highlighted_solution = self.results[(self.results.c == highlight["c"]) &
                                                        (self.results.win == highlight["win"]) &
                                                        (self.results.ratio == highlight["ratio"])]
                    highlighted_solution = highlighted_solution.sort_values("iter")

                    plt.plot(highlighted_solution.loc[:, metrics[i]], highlighted_solution.loc[:, metrics[j]],  color="orange")

                plt.xlabel(metrics[i])
                plt.ylabel(metrics[j])

    def plot_solutions_groupby_iter(self, metrics, c=[0.5, 1., 2., 4., 8.], win=np.arange(10), ratio =np.linspace(0,0.9,10)):
        """
        plot all solutions given the objective (metrics) for every c, win, and ratio tuplets
        :param metrics: list of metrics (e.g., ["CG_EGA_AP","RMSE"]), min 2
        :param highlight: dict with keys "c", "win", "ratio"; used to highlight specific results
        :param c: list of coherence factors
        :param win: list of window sizes
        :param ratio: list of ratios
        :return:
        """
        if len(metrics) < 2:
            sys.exit()
        for i in range(len(metrics) - 1):
            for j in range(i + 1, len(metrics)):
                plt.figure()
                for c_ in c:
                    for win_ in win:
                        for ratio_ in ratio:
                            groupby = self.results[
                                (self.results.c == c_) & (self.results.win == win_) & (self.results.ratio == ratio_)]
                            groupby = groupby.sort_values(metrics[i])
                            label = "c" + str(c_) + "_win" + str(win_) + "_ratio" + str(ratio_)
                            plt.plot(groupby.loc[:, metrics[i]], groupby.loc[:, metrics[j]], label=label)
                plt.xlabel(metrics[i])
                plt.ylabel(metrics[j])
                plt.legend()

    def plot_pareto(self, metrics, x_label, y_label):
        """
        2d plot pareto solutions given x_label and y_label
        :param metrics: list of metrics from which pareto front is computed (e.g., ["RMSE","CG_EGA_AP"]
        :param x_label: name of x axis (e.g., "RMSE")
        :param y_label: name of y axis (e.g., "CG_EGA_AP")
        :return:
        """
        pareto = self.undominated_solutions(metrics)
        pareto.sort_values(x_label).plot(x_label, y_label, style="o")

    def undominated_solutions(self, metrics):
        """
        compute the pareto/undominated front given objectifs (metrics)
        :param metrics: list of objectives (e.g., ["RMSE","CG_EGA_AP"]
        :return:
        """
        # extract points in results given the chosen metrics
        results_min = self.results.copy()
        # transform metrics to be maximised in metrics to be minimized
        results_min.loc[:, self.max_metrics] = -results_min.loc[:, self.max_metrics]
        costs = results_min.loc[:, metrics].values
        pareto_mask = self.pareto_mask(costs, return_mask=True)
        return self.results[pareto_mask]

    def undominated_training(self, metrics):
        """
        compute the pareto/undominated front given objectifs (metrics)
        :param metrics: list of objectives (e.g., ["RMSE","CG_EGA_AP"]
        :return:
        """
        results_min = self.results.copy()
        # transform metrics to be maximised in metrics to be minimized
        results_min.loc[:, self.max_metrics] = -results_min.loc[:, self.max_metrics]
        costs = []
        df = pd.DataFrame(columns=["c","win","ratio","n_dominating_pts"])
        for c in results_min.c.unique():
            for win in results_min.win.unique():
                for ratio in results_min.ratio.unique():
                    costs.append(
                        results_min.groupby(["c", "win", "ratio"]).get_group((c, win, ratio)).loc[:, metrics].values)
                    df = df.append(pd.DataFrame(data=[[c,win,ratio,0]], columns=["c", "win", "ratio","n_dominating_pts"]))
        df.n_dominating_pts = self.pareto_training_mask(np.array(costs))
        return df
        # return np.array(costs)

    def pareto_training_mask(self, costs):
        """
        Compute the pareto mask (1 if pareto 0 if not) of array
        :param costs: of array of shape (samples, objectives)
        :return: list of dominating points
        """
        dominating_pts = []
        next_point_index = 0
        while next_point_index < len(costs):
            n_dominating_pts = 0
            for pts in costs.reshape(-1, costs.shape[2]):
                if not np.any(np.all(costs[next_point_index] > pts,axis=1),axis=0) : n_dominating_pts += 1
            dominating_pts.append(n_dominating_pts)
            next_point_index+=1
        return dominating_pts

    def pareto_mask(self, costs, return_mask=True):
        # source = https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
        """
        Find the pareto-efficient points
        :param costs: An (n_points, n_costs) array
        :param return_mask: True to return a mask
        :return: An array of indices of pareto-efficient points.
            If return_mask is True, this will be an (n_points, ) boolean array
            Otherwise it will be a (n_efficient_points, ) integer array of indices.
        """
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index < len(costs):
            nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype=bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient
