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
        return sklearn.train_test_split(covariates_X, treatment_Y, train_size=split_size, random_state=42)

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
    def get_dropout_probability(entropy, gama=1):
        return 1 - (gama * 0.5) - (entropy * 0.5)

    @staticmethod
    def get_dropout_mask(prob, x):
        return Bernoulli(torch.full_like(x, 1 - prob)).sample() / (1 - prob)

    @staticmethod
    def get_dropout_probability_tensor(entropy, gama=1):
        return 1 - (gama * 0.5) - (entropy * 0.5)

    @staticmethod
    def KL_divergence(rho, rho_hat, device):
        # sigmoid because we need the probability distributions
        rho_hat = torch.mean(torch.sigmoid(rho_hat), 1)
        rho = torch.tensor([rho] * len(rho_hat)).to(device)
        return torch.sum(rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat)))

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
