import torch.nn as nn
from torch.utils.data import TensorDataset
import os
import numpy as np
import torch
from processing.models.predictor import Predictor
from misc.utils import printd
import misc.constants as cs
from torch import Tensor

class DeepPredictor(Predictor):
    def __init__(self, subject, ph, params, train, valid, test):
        super().__init__(subject, ph, params, train, valid, test)
        self.checkpoint_file = self._compute_checkpoint_file(self.__class__.__name__)
        self.input_shape = self._compute_input_shape()

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))
        torch.save(self.model.state_dict(), self.checkpoint_file)

    def _to_tensor_ds(self, x, y):
        return TensorDataset(torch.Tensor(x).cuda(), torch.Tensor(y).cuda())

    def _clear_checkpoint(self):
        os.remove(self.checkpoint_file)

    def _compute_checkpoint_file(self, model_name):
        rnd = np.random.randint(1e7)
        checkpoint_file = os.path.join(cs.path, "tmp", "checkpoints", model_name + "_" + str(rnd) + ".pt")
        printd("Saved model's file:", checkpoint_file)
        return checkpoint_file

    def _compute_loss_func(self):
        loss_func = nn.MSELoss()
        return loss_func

    def _reshape(self, data):
        t = data["datetime"]
        y = data["y"].values

        g = data.loc[:, [col for col in data.columns if "glucose" in col]].values
        cho = data.loc[:, [col for col in data.columns if "CHO" in col]].values
        ins = data.loc[:, [col for col in data.columns if "insulin" in col]].values

        # reshape timeseties in (n_samples, hist, 1) shape and concatenate
        g = g.reshape(-1, g.shape[1], 1)
        cho = cho.reshape(-1, g.shape[1], 1)
        ins = ins.reshape(-1, g.shape[1], 1)
        x = np.concatenate([g, cho, ins], axis=2)

        return x, y, t

    def extract_features(self, dataset, file):
        x, y, _ = self._str2dataset(dataset)
        self.model.load_state_dict(torch.load(file))
        self.model.eval()
        features = self.model.encoder(Tensor(x).cuda()).detach().cpu().numpy()
        features = features.reshape(features.shape[0], -1)

        return [features, y]