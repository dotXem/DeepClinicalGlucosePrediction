import numpy as np
import pandas as pd
from os import path
import misc.constants as cs
import misc
import matplotlib.pyplot as plt
from preprocessing.preprocessing import preprocessing

def compute_glucose_distribution(dataset, train_valid_or_test="train", plot=False, save=False, hypo_hyper_stats=False):
    """ load data"""
    glucose = []
    for subject in misc.datasets.datasets[dataset]["subjects"]:
        glucose_sbj = []
        train, valid, test, scalers = preprocessing(dataset, subject, 30 // 5, 180 // 5, 1440 // 5)
        if train_valid_or_test == "train":
            set = train
        elif train_valid_or_test == "valid":
            set = valid
        elif train_valid_or_test == "test":
            set = test
        for set_i, scalers_i in zip(set, scalers):
            glucose_sbj.append(set_i.y.values * scalers_i.scale_[-1] + scalers_i.mean_[-1])
        glucose.append(glucose_sbj)

    """ create average subject histograms """
    nbins = 40
    n_sbj = []
    for glucose_sbj in glucose:
        n_split = []
        for glucose_sbj_split in glucose_sbj:
            (n, bins, _) = plt.hist(glucose_sbj_split, bins=nbins, range=[0, 400], density=True, stacked=True)
            plt.close()
            n_split.append(n)
        n_sbj.append(np.mean(n_split, axis=0))


    """ compute distributions """
    n_arr = np.array(n_sbj) * 400 / nbins
    middle_bins = ((bins[1:] + bins[:-1]) / 2)
    mean = np.mean(n_arr, axis=0)
    std = np.std(n_arr, axis=0)

    """ plot """
    if plot:
        plt.figure()
        plt.plot(middle_bins, mean, color='#CC4F1B')

        plt.fill_between(middle_bins, mean - std, mean + std,
                         alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        plt.title("Distribution des échantillons de glycémie pour le jeu de données " + dataset + ".")
        plt.xlabel("glycémie [mg/dL]")
        plt.ylabel("probabilité")

    """ save """
    if save :
        df = pd.DataFrame(data=np.c_[middle_bins, mean, mean - std, mean + std],
                          columns=["middle_bins", "mean", "plus-std", "minus-std"])
        df.to_csv(path.join(cs.path, "tmp", "figures_data", "glucose_distribution_" + dataset + "_" + train_valid_or_test + ".dat"),
                  index_label="index")

    """ hypo hyper stats """
    if hypo_hyper_stats:
        print(np.sum(mean[np.where(bins <= 70)[0][:-1]]) * 100)
        print(np.sum(mean[np.where(bins >= 180)[0][:-1]]) * 100)

    return n_arr, bins, middle_bins, mean, std
