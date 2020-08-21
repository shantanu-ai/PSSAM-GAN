import math
from collections import namedtuple
from itertools import product

import numpy as np
import pandas as pd
import sklearn.model_selection as sklearn
import torch
from torch.distributions import Bernoulli


class Utils:
    @staticmethod
    def convert_df_to_np_arr(data):
        return data.to_numpy()

    @staticmethod
    def convert_to_col_vector(np_arr):
        return np_arr.reshape(np_arr.shape[0], 1)

    @staticmethod
    def test_train_split(covariates_X, treatment_Y, split_size=0.8):
        return sklearn.train_test_split(covariates_X, treatment_Y, train_size=split_size)

    @staticmethod
    def convert_to_tensor(X, Y):
        tensor_x = torch.stack([torch.Tensor(i) for i in X])
        tensor_y = torch.from_numpy(Y)
        processed_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
        return processed_dataset

    @staticmethod
    def convert_to_tensor_DCN(X, ps_score, Y_f, Y_cf):
        tensor_x = torch.stack([torch.Tensor(i) for i in X])
        tensor_ps_score = torch.from_numpy(ps_score)
        tensor_y_f = torch.from_numpy(Y_f)
        tensor_y_cf = torch.from_numpy(Y_cf)
        processed_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_ps_score,
                                                           tensor_y_f, tensor_y_cf)
        return processed_dataset

    @staticmethod
    def concat_np_arr(X, Y, axis=1):
        return np.concatenate((X, Y), axis)

    @staticmethod
    def get_device():
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_num_correct(preds, labels):
        return preds.argmax(dim=1).eq(labels).sum().item()

    @staticmethod
    def get_shanon_entropy(prob):
        if prob < 0:
            return
        if prob == 1:
            return -(prob * math.log2(prob))
        elif prob == 0:
            return -((1 - prob) * math.log2(1 - prob))
        else:
            return -(prob * math.log2(prob)) - ((1 - prob) * math.log2(1 - prob))

    @staticmethod
    def get_shanon_entropy_tensor(prob):
        prob_one_indx = prob == 1
        prob[prob_one_indx] = 0.999

        prob_zero_indx = prob == 0
        prob[prob_zero_indx] = 0.0001
        return -(prob * torch.log2(prob)) - ((1 - prob) * torch.log2(1 - prob))

    @staticmethod
    def get_dropout_probability(entropy, gama=1):
        return 1 - (gama * 0.5) - (entropy * 0.5)

    @staticmethod
    def get_dropout_mask(prob, x):
        x_tensor = torch.empty(1, x.size(1), device=Utils.get_device())
        out_val = np.empty([0, x.size(1)], dtype=float)
        if prob.dim() == 1:
            for prob_v in prob:
                v = Bernoulli(torch.full_like(x_tensor, 1 - prob_v.item())).sample() / (1 - prob_v.item())
                v = v.cpu().numpy()
                out_val = np.concatenate((out_val, v), axis=0)
                return torch.from_numpy(out_val).to(Utils.get_device())
        else:
            return Bernoulli(torch.full_like(x_tensor, 1 - prob.item())).sample() / (1 - prob.item())

    @staticmethod
    def get_dropout_mask_constant(prob, x):
        return Bernoulli(torch.full_like(x, 1 - prob)).sample() / (1 - prob)

    @staticmethod
    def get_dropout_probability_tensor(entropy, gama=1):
        return 1 - (gama * 0.5) - (entropy * 0.5)

    @staticmethod
    def get_runs(params):
        """
        Gets the run parameters using cartesian products of the different parameters.
        :param params: different parameters like batch size, learning rates
        :return: iterable run set
        """
        Run = namedtuple("Run", params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs

    @staticmethod
    def write_to_csv(file_name, list_to_write):
        pd.DataFrame.from_dict(
            list_to_write,
            orient='columns'
        ).to_csv(file_name)

    @staticmethod
    def create_tensors_to_train_DCN(group, dL):
        np_df_X = group[0]
        np_ps_score = group[1]
        np_df_Y_f = group[2]
        np_df_Y_cf = group[3]
        tensor = Utils.convert_to_tensor_DCN(np_df_X, np_ps_score,
                                             np_df_Y_f, np_df_Y_cf)
        return tensor

    @staticmethod
    def create_tensors_from_tuple(group):
        np_df_X = group[0]
        np_ps_score = group[1]
        np_df_Y_f = group[2]
        np_df_Y_cf = group[3]
        tensor = Utils.convert_to_tensor_DCN(np_df_X, np_ps_score,
                                             np_df_Y_f, np_df_Y_cf)
        return tensor

    @staticmethod
    def convert_to_tensor_DCN_PS(tensor_x, ps_score):
        tensor_ps_score = torch.from_numpy(ps_score)
        processed_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_ps_score)
        return processed_dataset

    @staticmethod
    def convert_to_tensor_DCN_semi_supervised(X, ps_score, T, Y_f, Y_cf):
        tensor_x = torch.stack([torch.Tensor(i) for i in X])
        tensor_ps_score = torch.from_numpy(ps_score)
        tensor_T = torch.from_numpy(T)
        tensor_y_f = torch.from_numpy(Y_f)
        tensor_y_cf = torch.from_numpy(Y_cf)
        processed_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_ps_score, tensor_T,
                                                           tensor_y_f, tensor_y_cf)
        return processed_dataset

    @staticmethod
    def create_tensors_to_train_DCN_semi_supervised(group):
        np_df_X = group[0]
        np_ps_score = group[1]
        T = group[2]
        np_df_Y_f = group[3]
        np_df_Y_cf = group[4]
        tensor = Utils.convert_to_tensor_DCN_semi_supervised(np_df_X, np_ps_score, T,
                                                             np_df_Y_f, np_df_Y_cf)
        return tensor

class EarlyStopping_DCN:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0,
                 model_shared_path='shared_checkpoint.pt',
                 model_y1_path='y1_checkpoint.pt',
                 model_y0_path='y0_checkpoint.pt',
                 trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.shared_path = model_shared_path
        self.model_y1_path = model_y1_path
        self.model_y0_path = model_y0_path
        self.trace_func = trace_func

    def __call__(self, val_loss, shared_model, y1_model, y0_model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, shared_model, y1_model, y0_model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, shared_model, y1_model, y0_model)
            self.counter = 0

    def save_checkpoint(self, val_loss, shared_model, y1_model, y0_model):
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min} --> {val_loss}).  Saving model ...')
        torch.save(shared_model.state_dict(), self.shared_path)
        torch.save(y1_model.state_dict(), self.model_y1_path)
        torch.save(y0_model.state_dict(), self.model_y0_path)
        self.val_loss_min = val_loss
