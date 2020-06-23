import pandas as pd
import numpy as np
from processing.models.pytorch_tools.losses import cMSE, gcMSE
import torch
import os
from processing.models.deep_predictor import DeepPredictor
import torch.nn as nn
from processing.models.pytorch_tools.training import fit, predict_double_y
import misc.constants as cs

class PCLSTM(DeepPredictor):
    def __init__(self, subject, ph, params, train, valid, test):
        super().__init__(subject, ph, params, train, valid, test)

        self.model = self.LSTM_Module(self.input_shape, self.params["hidden"], self.params["dropout_weights"],
                                      self.params["dropout_layer"])
        self.model.cuda()

    def fit(self, mean, std):
        x_train, y_train, t_train = self._str2dataset("train")
        x_valid, y_valid, t_valid = self._str2dataset("valid")
        train_ds = self._to_tensor_ds(x_train, y_train)
        valid_ds = self._to_tensor_ds(x_valid, y_valid)

        self.loss_func = gcMSE(mean, std, cs.freq, self.params["p_coeff"], self.params["r_coeff"], self.params["coherence_factor"])

        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.params["lr"], weight_decay=self.params["l2"])

        fit(self.params["epochs"], self.params["batch_size"], self.model, self.loss_func, self.opt, train_ds, valid_ds,
            self.params["patience"], self.checkpoint_file)

    def predict(self, dataset, clear=True):
        # get the data for which we make the predictions
        x, y, t = self._str2dataset(dataset)
        ds = self._to_tensor_ds(x, y)

        # create the model
        self.model.load_state_dict(torch.load(self.checkpoint_file))

        y_true, y_pred = predict_double_y(self.model, ds)
        results = self._format_results(y_true, y_pred, t)

        if clear:
            self._clear_checkpoint()

        return results

    def save(self, save_file):
        self.model.load_state_dict(torch.load(self.checkpoint_file))
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        torch.save(self.model.state_dict(), save_file)

    def _compute_input_shape(self):
        x_train, _, _ = self._str2dataset("train")
        return x_train.shape[2]

    def _reshape(self, data):
        x, y, t = super()._reshape(data)
        y_prev = data.y_prev.values
        return x, np.c_[y_prev, y], t

    def _format_results(self, y_true, y_pred, t):
        return pd.DataFrame(data=np.c_[y_true[:, 1], y_pred[:, 1]], index=pd.DatetimeIndex(t.values),
                            columns=["y_true", "y_pred"])

    class LSTM_Module(nn.Module):

        def __init__(self, n_in, neurons, dropout_weights, dropout_layer):
            super().__init__()

            self.encoder = nn.LSTM(n_in, neurons[0], len(neurons), dropout=dropout_layer, batch_first=True)
            self.regressor = nn.Linear(neurons[-1], 1)

        def forward(self, xb):
            features, _ = self.encoder(xb)
            prediction = self.regressor(features[:, -2:])
            return prediction.reshape((-1,2))
