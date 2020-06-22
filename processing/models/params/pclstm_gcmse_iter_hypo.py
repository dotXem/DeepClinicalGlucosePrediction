from postprocessing.smoothing import *
import numpy as np

def update_rule(iter):
    return 0.9 ** iter

parameters = {
    "hist": 180,

    "loss_func": "gcMSE",
    "coherence_factor": 8,
    "p_coeff": {"A": 1., "B": 1., "uC": 1., "lC": 1., "uD": 10., "lD": 1., "uE": 10., "lE": 1.},
    "r_coeff": {"A": 0., "B": 0., "uC": 1., "lC": 1., "uD": 1., "lD": 1., "uE": 1., "lE": 1.},

    # model hyperparameters
    "hidden": [256, 256],

    # training_old hyperparameters
    "dropout_weights": 0.0,
    "dropout_layer": 0.0,
    "epochs": 5000,
    "batch_size": 50,
    "lr": 5e-5,
    "l2": 1e-4,
    "patience": 10,

    # iterative cgega training
    "criteria": None,

    # smoothing
    # "smoothing": {"func": moving_average, "params": [5, 0.4]},  # OhioT1DM
    "smoothing": {"func": exponential_smoothing, "params": [0.85]},  # OhioT1DM
    # "smoothing": {"func": moving_average, "params": [3, 0.2]},  # IDIAB
#    "smoothing": {"func": exponential_smoothing, "params": [0.85]},  # IDIAB

    # p_ab update rule
    "p_ab_update": update_rule,
    # "p_ab_update": lambda iter: 1 / (iter + 1),
}


search = {
}


