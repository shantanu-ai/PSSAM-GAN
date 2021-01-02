from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from PSM_Manager import PSM_Manager
from TARNet_Model import TARNetH_Y1, TARNetH_Y0, TARNetPhi
from Utils import EarlyStopping_DCN


class TARNet_Manager:
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

    def train_semi_supervised(self, train_parameters, val_parameters, n_total, n_treated, device):
        epochs = train_parameters["epochs"]
        batch_size = train_parameters["batch_size"]
        lr = train_parameters["lr"]
        weight_decay = train_parameters["lambda"]
        shuffle = train_parameters["shuffle"]
        tensor_dataset = train_parameters["tensor_dataset"]

        treated_set_val = val_parameters["treated_set"]
        control_set_val = val_parameters["control_set"]

        u = n_treated / n_total
        weight_t = 1 / (2 * u)
        weight_c = 1 / (2 * (1 - u))

        treated_data_loader_val = torch.utils.data.DataLoader(treated_set_val,
                                                              shuffle=False)
        control_data_loader_val = torch.utils.data.DataLoader(control_set_val,
                                                              shuffle=False)

        treated_data_loader_train = torch.utils.data.DataLoader(tensor_dataset,
                                                                batch_size=batch_size,
                                                                shuffle=shuffle)

        optimizer_W = optim.Adam(self.tarnet_phi.parameters(), lr=lr)
        optimizer_V1 = optim.Adam(self.tarnet_h_y1.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer_V2 = optim.Adam(self.tarnet_h_y0.parameters(), lr=lr, weight_decay=weight_decay)

        lossF = nn.MSELoss()
        train_losses = []
        valid_losses = []

        early_stopping = EarlyStopping_DCN(patience=200, verbose=True,
                                           model_shared_path="Tarnet_shared_checkpoint.pt",
                                           model_y1_path="Tarnet_y1_checkpoint.pt",
                                           model_y0_path="Tarnet_y0_checkpoint.pt")
        for epoch in range(epochs):
            epoch += 1
            total_loss_T_train = 0
            total_loss_C_train = 0
            total_loss_T_val = 0
            total_loss_C_val = 0

            self.tarnet_phi.train()
            self.tarnet_h_y1.train()
            self.tarnet_h_y0.train()

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
                    total_loss_T_train += loss_T.item()

                if control_size > 0:
                    y0_hat = self.tarnet_h_y0(self.tarnet_phi(covariates_X_control))
                    if torch.cuda.is_available():
                        loss_C = weight_c * lossF(y0_hat.float().cuda(),
                                                  y_f_control.float().cuda()).to(device)
                    else:
                        loss_C = weight_c * lossF(y0_hat.float(),
                                                  y_f_control.float()).to(device)
                    loss_C.backward()
                    total_loss_C_train += loss_C.item()

                train_loss = total_loss_T_train + total_loss_C_train
                train_losses.append(train_loss)

                optimizer_W.step()

                if treated_size > 0:
                    optimizer_V1.step()
                if control_size > 0:
                    optimizer_V2.step()

            ######################
            # validate the model #
            ######################
            # prep model for evaluation
            self.tarnet_phi.eval()
            self.tarnet_h_y1.eval()
            self.tarnet_h_y0.eval()

            # val treated
            for batch in treated_data_loader_val:
                covariates_X, ps_score, y_f, y_cf = batch
                covariates_X = covariates_X.to(device)
                ps_score = ps_score.squeeze().to(device)
                y1_hat = self.tarnet_h_y1(self.tarnet_phi(covariates_X))
                if torch.cuda.is_available():
                    loss = lossF(y1_hat.float().cuda(),
                                 y_f.float().cuda()).to(device)
                else:
                    loss = lossF(y1_hat.float(),
                                 y_f.float()).to(device)
                total_loss_T_val += loss.item()


            # val control
            for batch in control_data_loader_val:
                covariates_X, ps_score, y_f, y_cf = batch
                covariates_X = covariates_X.to(device)
                ps_score = ps_score.squeeze().to(device)

                y0_hat = self.tarnet_h_y0(self.tarnet_phi(covariates_X))
                if torch.cuda.is_available():
                    loss = lossF(y0_hat.float().cuda(),
                                 y_f.float().cuda()).to(device)
                else:
                    loss = lossF(y0_hat.float(),
                                 y_f.float()).to(device)
                total_loss_C_val += loss.item()

            val_loss = total_loss_T_val + total_loss_C_val
            valid_losses.append(val_loss)

            train_loss = np.average(np.array(train_losses))
            valid_loss = np.average(np.array(valid_losses))

            train_losses = []
            valid_losses = []
            early_stopping(valid_loss, self.tarnet_phi, self.tarnet_h_y1, self.tarnet_h_y0)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            if epoch % 100 == 0:
                print("---->>>[[epoch: {0}/3000]], Treated + Control loss, train: {1}".format(epoch,
                                                                                              train_loss))
        self.tarnet_phi.load_state_dict(torch.load("Tarnet_shared_checkpoint.pt"))
        self.tarnet_h_y1.load_state_dict(torch.load("Tarnet_y1_checkpoint.pt"))
        self.tarnet_h_y0.load_state_dict(torch.load("Tarnet_y0_checkpoint.pt"))

    def eval_semi_supervised(self, eval_parameters, device, treated_flag):
        eval_set = eval_parameters["tensor_dataset"]

        _data_loader = torch.utils.data.DataLoader(eval_set,
                                                   shuffle=False)

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
                                                                shuffle=shuffle)

        optimizer_W = optim.Adam(self.tarnet_phi.parameters(), lr=lr)
        optimizer_V1 = optim.Adam(self.tarnet_h_y1.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer_V2 = optim.Adam(self.tarnet_h_y0.parameters(), lr=lr, weight_decay=weight_decay)
        lossF = nn.MSELoss()
        for epoch in range(epochs):
            epoch += 1
            total_loss_T = 0
            total_loss_C = 0
            for batch in treated_data_loader_train:
                covariates_X_treated, ps_score_treated, y_f_treated, y_cf_treated = batch
                covariates_X_treated = covariates_X_treated.to(device)
                ps_score_treated = ps_score_treated.squeeze().to(device)

                _tuple_treated = self.get_np_tuple_from_tensor(covariates_X_treated.cpu(),
                                                               ps_score_treated.cpu(),
                                                               y_f_treated.cpu(),
                                                               y_cf_treated.cpu())
                psm = PSM_Manager()
                tuple_control_dict = psm.match_using_prop_score(_tuple_treated, tuple_control)

                tuple_matched_control = tuple_control_dict["tuple_matched_control"]
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
                                                          shuffle=False)
        control_data_loader = torch.utils.data.DataLoader(control_set,
                                                          shuffle=False)

        err_treated_list = []
        err_control_list = []
        true_ITE_list = []
        predicted_ITE_list = []

        ITE_dict_list = []

        y1_true_list = []
        y1_hat_list = []

        y0_true_list = []
        y0_hat_list = []

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

            ITE_dict_list.append(self.create_ITE_Dict(covariates_X,
                                                      ps_score.item(), y_f.item(),
                                                      y_cf.item(),
                                                      true_ITE.item(),
                                                      predicted_ITE.item(),
                                                      diff.item()))

            y1_true_list.append(y_f.item())
            y0_true_list.append(y_cf.item())
            y1_hat_list.append(y1_hat.item())
            y0_hat_list.append(y0_hat.item())

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

            ITE_dict_list.append(self.create_ITE_Dict(covariates_X,
                                                      ps_score.item(), y_f.item(),
                                                      y_cf.item(),
                                                      true_ITE.item(),
                                                      predicted_ITE.item(),
                                                      diff.item()))
            y1_true_list.append(y_cf.item())
            y0_true_list.append(y_f.item())
            y1_hat_list.append(y1_hat.item())
            y0_hat_list.append(y0_hat.item())

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
            "ITE_dict_list": ITE_dict_list,

            "y1_true_list": y1_true_list,
            "y0_true_list": y0_true_list,
            "y1_hat_list": y1_hat_list,
            "y0_hat_list": y0_hat_list
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
