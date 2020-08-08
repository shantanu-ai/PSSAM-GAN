from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data
from sklearn.neighbors import NearestNeighbors

from Utils import Utils


class PS_Matching:
    def match_using_prop_score(self, tuple_treated, tuple_control):
        matched_controls = []

        # do ps match
        np_treated_df_X, np_treated_ps_score, np_treated_df_Y_f, np_treated_df_Y_cf = tuple_treated
        np_control_df_X, np_control_ps_score, np_control_df_Y_f, np_control_df_Y_cf = tuple_control

        # get unmatched controls
        matched_control_indices, unmatched_control_indices = self.get_matched_and_unmatched_control_indices(
            Utils.convert_to_col_vector(np_treated_ps_score),
            Utils.convert_to_col_vector(np_control_ps_score))

        tuple_matched_control, tuple_unmatched_control = self.filter_matched_and_unmatched_control_samples(
            np_control_df_X, np_control_ps_score,
            np_control_df_Y_f,
            np_control_df_Y_cf, matched_control_indices,
            unmatched_control_indices)

        return tuple_matched_control

    def filter_matched_and_unmatched_control_samples(self, np_control_df_X, np_control_ps_score,
                                                     np_control_df_Y_f,
                                                     np_control_df_Y_cf, matched_control_indices,
                                                     unmatched_control_indices):
        tuple_matched_control = self.filter_control_groups(np_control_df_X, np_control_ps_score,
                                                           np_control_df_Y_f,
                                                           np_control_df_Y_cf,
                                                           matched_control_indices)

        tuple_unmatched_control = self.filter_control_groups(np_control_df_X, np_control_ps_score,
                                                             np_control_df_Y_f,
                                                             np_control_df_Y_cf,
                                                             unmatched_control_indices)

        return tuple_matched_control, tuple_unmatched_control

    @staticmethod
    def filter_control_groups(np_control_df_X, np_control_ps_score,
                              np_control_df_Y_f,
                              np_control_df_Y_cf, indices):
        np_filter_control_df_X = np.take(np_control_df_X, indices, axis=0)
        np_filter_control_ps_score = np.take(np_control_ps_score, indices, axis=0)
        np_filter_control_df_Y_f = np.take(np_control_df_Y_f, indices, axis=0)
        np_filter_control_df_Y_cf = np.take(np_control_df_Y_cf, indices, axis=0)
        tuple_matched_control = (np_filter_control_df_X, np_filter_control_ps_score,
                                 np_filter_control_df_Y_f, np_filter_control_df_Y_cf)

        return tuple_matched_control

    @staticmethod
    def get_matched_and_unmatched_control_indices(ps_treated, ps_control):
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(ps_control)
        distance, matched_control = nn.kneighbors(ps_treated)
        matched_control_indices = np.array(matched_control).ravel()

        # remove duplicates
        # matched_control_indices = list(dict.fromkeys(matched_control_indices))
        set_matched_control_indices = set(matched_control_indices)
        total_indices = list(range(len(ps_control)))
        unmatched_control_indices = list(filter(lambda x: x not in set_matched_control_indices,
                                                total_indices))

        return matched_control_indices, unmatched_control_indices

    @staticmethod
    def get_unmatched_prop_list(tensor_unmatched_control):
        control_data_loader_train = torch.utils.data.DataLoader(tensor_unmatched_control,
                                                                batch_size=1,
                                                                shuffle=False,
                                                                num_workers=1)
        ps_unmatched_control_list = []
        for batch in control_data_loader_train:
            covariates_X, ps_score, y_f, y_cf = batch
            ps_unmatched_control_list.append(ps_score.item())

        return ps_unmatched_control_list


##########################################
class TARNetPhi(nn.Module):
    def __init__(self, input_nodes, shared_nodes=200):
        super(TARNetPhi, self).__init__()

        # shared layer
        self.shared1 = nn.Linear(in_features=input_nodes, out_features=shared_nodes)
        nn.init.xavier_uniform_(self.shared1.weight)
        nn.init.zeros_(self.shared1.bias)

        self.shared2 = nn.Linear(in_features=shared_nodes, out_features=shared_nodes)
        nn.init.xavier_uniform_(self.shared2.weight)
        nn.init.zeros_(self.shared2.bias)

        self.shared3 = nn.Linear(in_features=shared_nodes, out_features=shared_nodes)
        nn.init.xavier_uniform_(self.shared3.weight)
        nn.init.zeros_(self.shared3.bias)

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.float().cuda()
        else:
            x = x.float()
        # shared layers
        x = F.relu(self.shared1(x))
        x = F.relu(self.shared2(x))
        x = F.relu(self.shared3(x))

        return x


class TARNetH_Y1(nn.Module):
    def __init__(self, input_nodes=200, outcome_nodes=100):
        super(TARNetH_Y1, self).__init__()

        # potential outcome1 Y(1)
        self.hidden1_Y1 = nn.Linear(in_features=input_nodes, out_features=outcome_nodes)
        nn.init.xavier_uniform_(self.hidden1_Y1.weight)
        nn.init.zeros_(self.hidden1_Y1.bias)

        self.hidden2_Y1 = nn.Linear(in_features=outcome_nodes, out_features=outcome_nodes)
        nn.init.xavier_uniform_(self.hidden2_Y1.weight)
        nn.init.zeros_(self.hidden2_Y1.bias)

        self.out_Y1 = nn.Linear(in_features=outcome_nodes, out_features=1)
        nn.init.xavier_uniform_(self.out_Y1.weight)
        nn.init.zeros_(self.out_Y1.bias)

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.float().cuda()
        else:
            x = x.float()

        # potential outcome1 Y(1)
        y1 = F.relu(self.hidden1_Y1(x))
        y1 = F.relu(self.hidden2_Y1(y1))
        y1 = self.out_Y1(y1)

        return y1


class TARNetH_Y0(nn.Module):
    def __init__(self, input_nodes=200, outcome_nodes=100):
        super(TARNetH_Y0, self).__init__()

        # potential outcome1 Y(0)
        self.hidden1_Y0 = nn.Linear(in_features=input_nodes, out_features=outcome_nodes)
        nn.init.xavier_uniform_(self.hidden1_Y0.weight)
        nn.init.zeros_(self.hidden1_Y0.bias)

        self.hidden2_Y0 = nn.Linear(in_features=outcome_nodes, out_features=outcome_nodes)
        nn.init.xavier_uniform_(self.hidden2_Y0.weight)
        nn.init.zeros_(self.hidden2_Y0.bias)

        self.out_Y0 = nn.Linear(in_features=outcome_nodes, out_features=1)
        nn.init.xavier_uniform_(self.out_Y0.weight)
        nn.init.zeros_(self.out_Y0.bias)

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.float().cuda()
        else:
            x = x.float()

        # potential outcome1 Y(0)
        y0 = F.relu(self.hidden1_Y0(x))
        y0 = F.relu(self.hidden2_Y0(y0))
        y0 = self.out_Y0(y0)

        return y0


#####################
class InferenceNet:
    def __init__(self, input_nodes, shared_nodes, outcome_nodes, device):
        self.tarnet_phi = TARNetPhi(input_nodes, shared_nodes=shared_nodes).to(device)

        self.tarnet_h_y1 = TARNetH_Y1(input_nodes=shared_nodes,
                                      outcome_nodes=outcome_nodes).to(device)

        self.tarnet_h_y0 = TARNetH_Y0(input_nodes=shared_nodes,
                                      outcome_nodes=outcome_nodes).to(device)

    def get_tarnet_phi(self):
        return self.tarnet_phi

    def get_tarnet_h_y1(self):
        return self.tarnet_h_y1

    def get_tarnet_h_y0_model(self):
        return self.tarnet_h_y0

    def train_semi_supervised(self, train_parameters, n_total, n_treated, device):
        epochs = train_parameters["epochs"]
        batch_size = train_parameters["batch_size"]
        lr = train_parameters["lr"]
        weight_decay = train_parameters["lambda"]
        shuffle = train_parameters["shuffle"]
        tensor_dataset = train_parameters["tensor_dataset"]
        u = n_treated / n_total
        weight_t = 1 / (2 * u)
        weight_c = 1 / (2 * (1 - u))

        treated_data_loader_train = torch.utils.data.DataLoader(tensor_dataset,
                                                                batch_size=batch_size,
                                                                shuffle=shuffle,
                                                                num_workers=1)

        optimizer_W = optim.Adam(self.tarnet_phi.parameters(), lr=lr)
        optimizer_V1 = optim.Adam(self.tarnet_h_y1.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer_V2 = optim.Adam(self.tarnet_h_y0.parameters(), lr=lr, weight_decay=weight_decay)
        lossF = nn.MSELoss()
        print(".. Training started ..")
        print(device)
        for epoch in range(epochs):
            epoch += 1
            total_loss_T = 0
            total_loss_C = 0
            for batch in treated_data_loader_train:
                covariates_X, ps_score, T, y_f, y_cf = batch
                covariates_X = covariates_X.to(device)
                ps_score = ps_score.squeeze().to(device)

                idx = (T == 1)
                covariates_X_treated = covariates_X[idx]
                y_f_treated = y_f[idx]
                covariates_X_control = covariates_X[~idx]
                y_f_control = y_f[~idx]

                treated_size = covariates_X_treated.size(0)
                control_size = covariates_X_control.size(0)

                optimizer_W.zero_grad()
                optimizer_V1.zero_grad()
                optimizer_V2.zero_grad()

                if treated_size > 0:
                    y1_hat = self.tarnet_h_y1(self.tarnet_phi(covariates_X_treated))
                    if torch.cuda.is_available():
                        loss_T = weight_t * lossF(y1_hat.float().cuda(),
                                                  y_f_treated.float().cuda()).to(device)
                    else:
                        loss_T = weight_t * lossF(y1_hat.float(),
                                                  y_f_treated.float()).to(device)
                    loss_T.backward()
                    total_loss_T += loss_T.item()

                if control_size > 0:
                    y0_hat = self.tarnet_h_y0(self.tarnet_phi(covariates_X_control))
                    if torch.cuda.is_available():
                        loss_C = weight_c * lossF(y0_hat.float().cuda(),
                                                  y_f_control.float().cuda()).to(device)
                    else:
                        loss_C = weight_c * lossF(y0_hat.float(),
                                                  y_f_control.float()).to(device)
                    loss_C.backward()
                    total_loss_C += loss_C.item()



                optimizer_W.step()

                if treated_size > 0:
                    optimizer_V1.step()
                if control_size > 0:
                    optimizer_V2.step()

            if epoch % 100 == 0:
                print("epoch: {0}, Treated + Control loss: {1}".format(epoch, total_loss_T + total_loss_C))

    def eval_semi_supervised(self, eval_parameters, device, treated_flag):
        eval_set = eval_parameters["tensor_dataset"]

        _data_loader = torch.utils.data.DataLoader(eval_set,
                                                   shuffle=False, num_workers=1)

        y_f_list = []
        y_cf_list = []

        for batch in _data_loader:
            covariates_X, ps_score = batch
            covariates_X = covariates_X.to(device)
            ps_score = ps_score.squeeze().to(device)
            y1_hat = self.tarnet_h_y1(self.tarnet_phi(covariates_X))
            y0_hat = self.tarnet_h_y0(self.tarnet_phi(covariates_X))
            if treated_flag:
                y_f_list.append(y1_hat.item())
                y_cf_list.append(y0_hat.item())
            else:
                y_f_list.append(y0_hat.item())
                y_cf_list.append(y1_hat.item())

        return {
            "y_f_list": np.array(y_f_list),
            "y_cf_list": np.array(y_cf_list)
        }

    def train(self, train_parameters, device):
        epochs = train_parameters["epochs"]
        batch_size = train_parameters["batch_size"]
        lr = train_parameters["lr"]
        weight_decay = train_parameters["lambda"]
        shuffle = train_parameters["shuffle"]
        treated_tensor_dataset = train_parameters["treated_tensor_dataset"]
        tuple_control = train_parameters["tuple_control_train"]

        treated_data_loader_train = torch.utils.data.DataLoader(treated_tensor_dataset,
                                                                batch_size=batch_size,
                                                                shuffle=shuffle,
                                                                num_workers=1)

        optimizer_W = optim.Adam(self.tarnet_phi.parameters(), lr=lr)
        optimizer_V1 = optim.Adam(self.tarnet_h_y1.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer_V2 = optim.Adam(self.tarnet_h_y0.parameters(), lr=lr, weight_decay=weight_decay)
        lossF = nn.MSELoss()
        print(".. Training started ..")
        print(device)
        for epoch in range(epochs):
            epoch += 1
            total_loss_T = 0
            total_loss_C = 0
            for batch in treated_data_loader_train:
                covariates_X_treated, ps_score_treated, y_f_treated, y_cf_treated = batch
                covariates_X_treated = covariates_X_treated.to(device)
                ps_score_treated = ps_score_treated.squeeze().to(device)

                _tuple_treated = self.get_np_tuple_from_tensor(covariates_X_treated, ps_score_treated,
                                                               y_f_treated, y_cf_treated)
                psm = PS_Matching()
                tuple_matched_control = psm.match_using_prop_score(_tuple_treated, tuple_control)

                covariates_X_control, ps_score_control, y_f_control, y_cf_control = \
                    self.get_tensor_from_np_tuple(tuple_matched_control)

                y1_hat = self.tarnet_h_y1(self.tarnet_phi(covariates_X_treated))
                y0_hat = self.tarnet_h_y0(self.tarnet_phi(covariates_X_control))

                if torch.cuda.is_available():
                    loss_T = lossF(y1_hat.float().cuda(),
                                   y_f_treated.float().cuda()).to(device)
                    loss_C = lossF(y0_hat.float().cuda(),
                                   y_f_control.float().cuda()).to(device)
                else:
                    loss_T = lossF(y1_hat.float(),
                                   y_f_treated.float()).to(device)
                    loss_C = lossF(y0_hat.float(),
                                   y_f_control.float()).to(device)

                optimizer_W.zero_grad()
                optimizer_V1.zero_grad()
                optimizer_V2.zero_grad()
                loss_T.backward()
                loss_C.backward()
                optimizer_W.step()
                optimizer_V1.step()
                optimizer_V2.step()
                total_loss_T += loss_T.item()
                total_loss_C += loss_C.item()

            if epoch % 100 == 0:
                print("epoch: {0}, Treated + Control loss: {1}".format(epoch, total_loss_T + total_loss_C))

    def eval(self, eval_parameters, device):
        treated_set = eval_parameters["treated_set"]
        control_set = eval_parameters["control_set"]
        treated_data_loader = torch.utils.data.DataLoader(treated_set,
                                                          shuffle=False, num_workers=1)
        control_data_loader = torch.utils.data.DataLoader(control_set,
                                                          shuffle=False, num_workers=1)

        err_treated_list = []
        err_control_list = []
        true_ITE_list = []
        predicted_ITE_list = []

        ITE_dict_list = []

        for batch in treated_data_loader:
            covariates_X, ps_score, y_f, y_cf = batch
            covariates_X = covariates_X.to(device)
            y1_hat = self.tarnet_h_y1(self.tarnet_phi(covariates_X))
            y0_hat = self.tarnet_h_y0(self.tarnet_phi(covariates_X))

            predicted_ITE = y1_hat - y0_hat
            true_ITE = y_f - y_cf

            if torch.cuda.is_available():
                diff = true_ITE.float().cuda() - predicted_ITE.float().cuda()
            else:
                diff = true_ITE.float() - predicted_ITE.float()

            # ITE_dict_list.append(self.create_ITE_Dict(covariates_X,
            #                                           ps_score.item(), y_f.item(),
            #                                           y_cf.item(),
            #                                           true_ITE.item(),
            #                                           predicted_ITE.item(),
            #                                           diff.item()))
            err_treated_list.append(diff.item())
            true_ITE_list.append(true_ITE.item())
            predicted_ITE_list.append(predicted_ITE.item())

        for batch in control_data_loader:
            covariates_X, ps_score, y_f, y_cf = batch
            covariates_X = covariates_X.to(device)

            y1_hat = self.tarnet_h_y1(self.tarnet_phi(covariates_X))
            y0_hat = self.tarnet_h_y0(self.tarnet_phi(covariates_X))

            predicted_ITE = y1_hat - y0_hat
            true_ITE = y_cf - y_f
            if torch.cuda.is_available():
                diff = true_ITE.float().cuda() - predicted_ITE.float().cuda()
            else:
                diff = true_ITE.float() - predicted_ITE.float()

            # ITE_dict_list.append(self.create_ITE_Dict(covariates_X,
            #                                           ps_score.item(), y_f.item(),
            #                                           y_cf.item(),
            #                                           true_ITE.item(),
            #                                           predicted_ITE.item(),
            #                                           diff.item()))
            err_control_list.append(diff.item())
            true_ITE_list.append(true_ITE.item())
            predicted_ITE_list.append(predicted_ITE.item())

        # print(err_treated_list)
        # print(err_control_list)
        return {
            "treated_err": err_treated_list,
            "control_err": err_control_list,
            "true_ITE": true_ITE_list,
            "predicted_ITE": predicted_ITE_list,
            "ITE_dict_list": ITE_dict_list
        }

    @staticmethod
    def get_np_tuple_from_tensor(covariates_X, ps_score, y_f, y_cf):
        np_covariates_X = covariates_X.numpy()
        ps_score = ps_score.numpy()
        y_f = y_f.numpy()
        y_cf = y_cf.numpy()
        _tuple = (np_covariates_X, ps_score, y_f, y_cf)

        return _tuple

    @staticmethod
    def get_tensor_from_np_tuple(_tuple):
        np_df_X, np_ps_score, np_df_Y_f, np_df_Y_cf = _tuple
        return torch.from_numpy(np_df_X), torch.from_numpy(np_ps_score), \
               torch.from_numpy(np_df_Y_f), torch.from_numpy(np_df_Y_cf),


############


from Constants import Constants


class DCN_shared(nn.Module):
    def __init__(self, input_nodes):
        super(DCN_shared, self).__init__()

        # shared layer
        self.shared1 = nn.Linear(in_features=input_nodes, out_features=200)
        nn.init.xavier_uniform_(self.shared1.weight)

        self.shared2 = nn.Linear(in_features=200, out_features=200)
        nn.init.xavier_uniform_(self.shared2.weight)

        self.dropout_2 = nn.Dropout(p=0.2)
        self.dropout_5 = nn.Dropout(p=0.5)
        self.training_mode = None

    def set_train_mode(self, training_mode):
        self.training_mode = training_mode

    def forward(self, x, ps_score):
        if torch.cuda.is_available():
            x = x.float().cuda()
        else:
            x = x.float()

        if self.training_mode == Constants.DCN_EVALUATION:
            x = self.__eval_net(x)
        elif self.training_mode == Constants.DCN_TRAIN_PD:
            x = self.__train_net_PD(x, ps_score=ps_score)
        elif self.training_mode == Constants.DCN_TRAIN_CONSTANT_DROPOUT_5:
            x = self.__train_net_constant_dropout(x, ps_score=0.5)
        elif self.training_mode == Constants.DCN_TRAIN_CONSTANT_DROPOUT_2:
            x = self.__train_net_constant_dropout(x, ps_score=0.2)
        elif self.training_mode == Constants.DCN_TRAIN_NO_DROPOUT:
            x = self.__train_net_no_droput(x)

        return x

    def __train_net_constant_dropout(self, x, ps_score):
        if ps_score == 0.2:
            drop_out = self.dropout_2
        elif ps_score == 0.5:
            drop_out = self.dropout_5

        # shared layers
        x = F.relu(drop_out(self.shared1(x)))
        x = F.relu(drop_out(self.shared2(x)))

        return x

    def __train_net_PD(self, x, ps_score):
        entropy = Utils.get_shanon_entropy(ps_score.item())
        dropout_prob = Utils.get_dropout_probability(entropy, gama=1)

        # shared layers
        shared_mask = Utils.get_dropout_mask(dropout_prob, self.shared1(x))
        x = F.relu(shared_mask * self.shared1(x))
        x = F.relu(shared_mask * self.shared2(x))

        return x

    def __train_net_no_droput(self, x):
        # shared layers
        x = F.relu(self.shared1(x))
        x = F.relu(self.shared2(x))
        return x

    def __eval_net(self, x):
        # shared layers
        x = F.relu(self.shared1(x))
        x = F.relu(self.shared2(x))

        return x


class DCN_Y1(nn.Module):
    def __init__(self):
        super(DCN_Y1, self).__init__()

        # potential outcome1 Y(1)
        self.hidden1_Y1 = nn.Linear(in_features=200, out_features=200)
        nn.init.xavier_uniform_(self.hidden1_Y1.weight)

        self.hidden2_Y1 = nn.Linear(in_features=200, out_features=200)
        nn.init.xavier_uniform_(self.hidden2_Y1.weight)

        self.out_Y1 = nn.Linear(in_features=200, out_features=1)
        nn.init.xavier_uniform_(self.out_Y1.weight)

        self.dropout_2 = nn.Dropout(p=0.2)
        self.dropout_5 = nn.Dropout(p=0.5)
        self.training_mode = None

    def set_train_mode(self, training_mode):
        self.training_mode = training_mode

    def forward(self, x, ps_score):
        if torch.cuda.is_available():
            x = x.float().cuda()
        else:
            x = x.float()

        if self.training_mode == Constants.DCN_EVALUATION:
            y1 = self.__eval_net(x)
        elif self.training_mode == Constants.DCN_TRAIN_PD:
            y1 = self.__train_net_PD(x, ps_score=ps_score)
        elif self.training_mode == Constants.DCN_TRAIN_CONSTANT_DROPOUT_5:
            y1 = self.__train_net_constant_dropout(x, ps_score=0.5)
        elif self.training_mode == Constants.DCN_TRAIN_CONSTANT_DROPOUT_2:
            y1 = self.__train_net_constant_dropout(x, ps_score=0.2)
        elif self.training_mode == Constants.DCN_TRAIN_NO_DROPOUT:
            y1 = self.__train_net_no_droput(x)

        return y1

    def __train_net_constant_dropout(self, x, ps_score):
        if ps_score == 0.2:
            drop_out = self.dropout_2
        elif ps_score == 0.5:
            drop_out = self.dropout_5

        # potential outcome1 Y(1)
        y1 = F.relu(drop_out(self.hidden1_Y1(x)))
        y1 = F.relu(drop_out(self.hidden2_Y1(y1)))
        y1 = self.out_Y1(y1)

        return y1

    def __train_net_PD(self, x, ps_score):
        entropy = Utils.get_shanon_entropy(ps_score.item())
        dropout_prob = Utils.get_dropout_probability(entropy, gama=1)

        # potential outcome1 Y(1)
        y1_mask = Utils.get_dropout_mask(dropout_prob, self.hidden1_Y1(x))
        y1 = F.relu(y1_mask * self.hidden1_Y1(x))
        y1 = F.relu(y1_mask * self.hidden2_Y1(y1))
        y1 = self.out_Y1(y1)

        return y1

    def __train_net_no_droput(self, x):
        # potential outcome1 Y(1)
        y1 = F.relu(self.hidden1_Y1(x))
        y1 = F.relu(self.hidden2_Y1(y1))
        y1 = self.out_Y1(y1)

        return y1

    def __eval_net(self, x):
        # potential outcome1 Y(1)
        y1 = F.relu(self.hidden1_Y1(x))
        y1 = F.relu(self.hidden2_Y1(y1))
        y1 = self.out_Y1(y1)

        return y1


class DCN_Y0(nn.Module):
    def __init__(self):
        super(DCN_Y0, self).__init__()

        # potential outcome1 Y(0)
        self.hidden1_Y0 = nn.Linear(in_features=200, out_features=200)
        nn.init.xavier_uniform_(self.hidden1_Y0.weight)

        self.hidden2_Y0 = nn.Linear(in_features=200, out_features=200)
        nn.init.xavier_uniform_(self.hidden2_Y0.weight)

        self.out_Y0 = nn.Linear(in_features=200, out_features=1)
        nn.init.xavier_uniform_(self.out_Y0.weight)

        self.dropout_2 = nn.Dropout(p=0.2)
        self.dropout_5 = nn.Dropout(p=0.5)
        self.training_mode = None

    def set_train_mode(self, training_mode):
        self.training_mode = training_mode

    def forward(self, x, ps_score):
        if torch.cuda.is_available():
            x = x.float().cuda()
        else:
            x = x.float()

        if self.training_mode == Constants.DCN_EVALUATION:
            y0 = self.__eval_net(x)
        elif self.training_mode == Constants.DCN_TRAIN_PD:
            y0 = self.__train_net_PD(x, ps_score=ps_score)
        elif self.training_mode == Constants.DCN_TRAIN_CONSTANT_DROPOUT_5:
            y0 = self.__train_net_constant_dropout(x, ps_score=0.5)
        elif self.training_mode == Constants.DCN_TRAIN_CONSTANT_DROPOUT_2:
            y0 = self.__train_net_constant_dropout(x, ps_score=0.2)
        elif self.training_mode == Constants.DCN_TRAIN_NO_DROPOUT:
            y0 = self.__train_net_no_droput(x)

        return y0

    def __train_net_constant_dropout(self, x, ps_score):
        if ps_score == 0.2:
            drop_out = self.dropout_2
        elif ps_score == 0.5:
            drop_out = self.dropout_5

        # potential outcome1 Y(0)
        y0 = F.relu(drop_out(self.hidden1_Y0(x)))
        y0 = F.relu(drop_out(self.hidden2_Y0(y0)))
        y0 = self.out_Y0(y0)

        return y0

    def __train_net_PD(self, x, ps_score):
        entropy = Utils.get_shanon_entropy(ps_score.item())
        dropout_prob = Utils.get_dropout_probability(entropy, gama=1)

        # potential outcome1 Y(0)
        y0_mask = Utils.get_dropout_mask(dropout_prob, self.hidden1_Y0(x))
        y0 = F.relu(y0_mask * self.hidden1_Y0(x))
        y0 = F.relu(y0_mask * self.hidden2_Y0(y0))
        y0 = self.out_Y0(y0)

        return y0

    def __train_net_no_droput(self, x):
        # potential outcome1 Y(0)
        y0 = F.relu(self.hidden1_Y0(x))
        y0 = F.relu(self.hidden2_Y0(y0))
        y0 = self.out_Y0(y0)

        return y0

    def __eval_net(self, x):
        # potential outcome1 Y(0)
        y0 = F.relu(self.hidden1_Y0(x))
        y0 = F.relu(self.hidden2_Y0(y0))
        y0 = self.out_Y0(y0)

        return y0


####################################

class DCN_network_2:
    def __init__(self, input_nodes, training_mode, device):
        self.dcn_shared = DCN_shared(input_nodes=input_nodes, ).to(device)
        self.dcn_y1 = DCN_Y1().to(device)
        self.dcn_y0 = DCN_Y0().to(device)

    def train(self, train_parameters, device, train_mode=Constants.DCN_TRAIN_PD):
        epochs = train_parameters["epochs"]
        treated_batch_size = train_parameters["treated_batch_size"]
        control_batch_size = train_parameters["control_batch_size"]
        lr = train_parameters["lr"]
        shuffle = train_parameters["shuffle"]
        treated_set_train = train_parameters["treated_set_train"]
        control_set_train = train_parameters["control_set_train"]

        self.dcn_shared.set_train_mode(training_mode=train_mode)
        self.dcn_y1.set_train_mode(training_mode=train_mode)
        self.dcn_y0.set_train_mode(training_mode=train_mode)

        treated_data_loader_train = torch.utils.data.DataLoader(treated_set_train,
                                                                batch_size=treated_batch_size,
                                                                shuffle=shuffle,
                                                                num_workers=1)

        control_data_loader_train = torch.utils.data.DataLoader(control_set_train,
                                                                batch_size=control_batch_size,
                                                                shuffle=shuffle,
                                                                num_workers=1)

        optimizer_shared = optim.Adam(self.dcn_shared.parameters(), lr=lr)
        optimizer_y1 = optim.Adam(self.dcn_y1.parameters(), lr=lr)
        optimizer_y0 = optim.Adam(self.dcn_y0.parameters(), lr=lr)
        lossF = nn.MSELoss()

        min_loss = 100000.0
        dataset_loss = 0.0
        print(".. Training started ..")
        print(device)
        for epoch in range(epochs):
            self.dcn_shared.train()
            self.dcn_y1.train()
            self.dcn_y0.train()
            total_loss = 0
            train_set_size = 0

            if epoch % 2 == 0:
                dataset_loss = 0
                # train treated
                for batch in treated_data_loader_train:
                    covariates_X, ps_score, y_f, y_cf = batch
                    covariates_X = covariates_X.to(device)
                    ps_score = ps_score.squeeze().to(device)
                    train_set_size += covariates_X.size(0)
                    y1_hat = self.dcn_y1(self.dcn_shared(covariates_X, ps_score), ps_score)
                    if torch.cuda.is_available():
                        loss = lossF(y1_hat.float().cuda(),
                                     y_f.float().cuda()).to(device)
                    else:
                        loss = lossF(y1_hat.float(),
                                     y_f.float()).to(device)

                    optimizer_shared.zero_grad()
                    optimizer_y1.zero_grad()
                    loss.backward()
                    optimizer_shared.step()
                    optimizer_y1.step()
                    total_loss += loss.item()
                dataset_loss = total_loss

            elif epoch % 2 == 1:
                # train control

                for batch in control_data_loader_train:
                    covariates_X, ps_score, y_f, y_cf = batch
                    covariates_X = covariates_X.to(device)
                    ps_score = ps_score.squeeze().to(device)

                    train_set_size += covariates_X.size(0)
                    y0_hat = self.dcn_y0(self.dcn_shared(covariates_X, ps_score), ps_score)
                    if torch.cuda.is_available():
                        loss = lossF(y0_hat.float().cuda(),
                                     y_f.float().cuda()).to(device)
                    else:
                        loss = lossF(y0_hat.float(),
                                     y_f.float()).to(device)
                    optimizer_shared.zero_grad()
                    optimizer_y0.zero_grad()
                    loss.backward()
                    optimizer_shared.step()
                    optimizer_y0.step()
                    total_loss += loss.item()
                    total_loss += loss.item()
                dataset_loss = dataset_loss + total_loss

            if epoch % 10 == 9:
                print("epoch: {0}, Treated + Control loss: {1}".format(epoch, dataset_loss))

    def eval(self, eval_parameters, device, input_nodes, train_mode):
        treated_set = eval_parameters["treated_set"]
        control_set = eval_parameters["control_set"]

        self.dcn_shared.eval()
        self.dcn_y1.eval()
        self.dcn_y0.eval()
        treated_data_loader = torch.utils.data.DataLoader(treated_set,
                                                          shuffle=False, num_workers=1)
        control_data_loader = torch.utils.data.DataLoader(control_set,
                                                          shuffle=False, num_workers=1)

        err_treated_list = []
        err_control_list = []
        true_ITE_list = []
        predicted_ITE_list = []

        ITE_dict_list = []

        for batch in treated_data_loader:
            covariates_X, ps_score, y_f, y_cf = batch
            covariates_X = covariates_X.to(device)
            ps_score = ps_score.squeeze().to(device)
            y1_hat = self.dcn_y1(self.dcn_shared(covariates_X, ps_score), ps_score)
            y0_hat = self.dcn_y0(self.dcn_shared(covariates_X, ps_score), ps_score)
            predicted_ITE = y1_hat - y0_hat

            true_ITE = y_f - y_cf
            if torch.cuda.is_available():
                diff = true_ITE.float().cuda() - predicted_ITE.float().cuda()
            else:
                diff = true_ITE.float() - predicted_ITE.float()

            ITE_dict_list.append(self.create_ITE_Dict(covariates_X,
                                                      ps_score.item(), y_f.item(),
                                                      y_cf.item(),
                                                      true_ITE.item(),
                                                      predicted_ITE.item(),
                                                      diff.item()))
            err_treated_list.append(diff.item())
            true_ITE_list.append(true_ITE.item())
            predicted_ITE_list.append(predicted_ITE.item())

        for batch in control_data_loader:
            covariates_X, ps_score, y_f, y_cf = batch
            covariates_X = covariates_X.to(device)
            ps_score = ps_score.squeeze().to(device)
            y1_hat = self.dcn_y1(self.dcn_shared(covariates_X, ps_score), ps_score)
            y0_hat = self.dcn_y0(self.dcn_shared(covariates_X, ps_score), ps_score)
            predicted_ITE = y1_hat - y0_hat
            true_ITE = y_cf - y_f
            if torch.cuda.is_available():
                diff = true_ITE.float().cuda() - predicted_ITE.float().cuda()
            else:
                diff = true_ITE.float() - predicted_ITE.float()

            ITE_dict_list.append(self.create_ITE_Dict(covariates_X,
                                                      ps_score.item(), y_f.item(),
                                                      y_cf.item(),
                                                      true_ITE.item(),
                                                      predicted_ITE.item(),
                                                      diff.item()))
            err_control_list.append(diff.item())
            true_ITE_list.append(true_ITE.item())
            predicted_ITE_list.append(predicted_ITE.item())

        # print(err_treated_list)
        # print(err_control_list)
        return {
            "treated_err": err_treated_list,
            "control_err": err_control_list,
            "true_ITE": true_ITE_list,
            "predicted_ITE": predicted_ITE_list,
            "ITE_dict_list": ITE_dict_list
        }

    def eval_semi_supervised(self, eval_parameters, device, input_nodes, train_mode, treated_flag):
        eval_set = eval_parameters["eval_set"]

        self.dcn_shared.eval()
        self.dcn_y1.eval()
        self.dcn_y0.eval()
        treated_data_loader = torch.utils.data.DataLoader(eval_set,
                                                          shuffle=False, num_workers=1)

        y_f_list = []
        y_cf_list = []

        for batch in treated_data_loader:
            covariates_X, ps_score = batch
            covariates_X = covariates_X.to(device)
            ps_score = ps_score.squeeze().to(device)
            y1_hat = self.dcn_y1(self.dcn_shared(covariates_X, ps_score), ps_score)
            y0_hat = self.dcn_y0(self.dcn_shared(covariates_X, ps_score), ps_score)
            if treated_flag:
                y_f_list.append(y1_hat.item())
                y_cf_list.append(y0_hat.item())
            else:
                y_f_list.append(y0_hat.item())
                y_cf_list.append(y1_hat.item())

        return {
            "y_f_list": np.array(y_f_list),
            "y_cf_list": np.array(y_cf_list)
        }

    @staticmethod
    def create_ITE_Dict(covariates_X, ps_score, y_f, y_cf, true_ITE,
                        predicted_ITE, diff):
        result_dict = OrderedDict()
        covariate_list = [element.item() for element in covariates_X.flatten()]
        idx = 0
        for item in covariate_list:
            idx += 1
            result_dict["X" + str(idx)] = item

        result_dict["ps_score"] = ps_score
        result_dict["factual"] = y_f
        result_dict["counter_factual"] = y_cf
        result_dict["true_ITE"] = true_ITE
        result_dict["predicted_ITE"] = predicted_ITE
        result_dict["diff"] = diff

        return result_dict
