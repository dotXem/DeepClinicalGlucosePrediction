from postprocessing.smoothing import *
import numpy as np

parameters = {
    "hist": 180,

    "loss_func": "gcMSE",
    "coherence_factor": 8,
    "p_coeff": {"A": 0., "B": 0., "uC": 1., "lC": 1., "uD": 5., "lD": 1., "uE": 5., "lE": 1.},
    "r_coeff": {"A": 0., "B": 0., "uC": 1., "lC": 1., "uD": 1., "lD": 1., "uE": 1., "lE": 1.},

    # model hyperparameters
    "hidden": [256, 256],

    # training_old hyperparameters
    "dropout_weights": 0.0,
    "dropout_layer": 0.0,
    "epochs": 2,
    "batch_size": 50,
    "lr": 5e-4,
    "l2": 1e-4,
    "patience": 50,
}

search = {
}
