from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn

from DCN_Model import DCN_shared, DCN_Y1, DCN_Y0
from Utils import EarlyStopping_DCN


class DCN_Manager:
    def __init__(self, input_nodes, device,
                 model_shared_path="",
                 model_y1_path="",
                 model_y0_path=""):
        self.dcn_shared = DCN_shared(input_nodes=input_nodes).to(device)
        self.dcn_y1 = DCN_Y1().to(device)
        self.dcn_y0 = DCN_Y0().to(device)
        self.model_shared_path = model_shared_path
        self.model_y1_path = model_y1_path
        self.model_y0_path = model_y0_path

    def train(self, train_parameters, val_parameters, device, train_mode, ss=False):
        epochs = train_parameters["epochs"]
        treated_batch_size = train_parameters["treated_batch_size"]
        control_batch_size = train_parameters["control_batch_size"]
        lr = train_parameters["lr"]
        shuffle = train_parameters["shuffle"]
        treated_set_train = train_parameters["treated_set_train"]
        control_set_train = train_parameters["control_set_train"]

        treated_set_val = val_parameters["treated_set"]
        control_set_val = val_parameters["control_set"]

        self.dcn_shared.set_train_mode(training_mode=train_mode)
        self.dcn_y1.set_train_mode(training_mode=train_mode)
        self.dcn_y0.set_train_mode(training_mode=train_mode)

        treated_data_loader_train = torch.utils.data.DataLoader(treated_set_train,
                                                                batch_size=treated_batch_size,
                                                                shuffle=shuffle)

        control_data_loader_train = torch.utils.data.DataLoader(control_set_train,
                                                                batch_size=control_batch_size,
                                                                shuffle=shuffle)

        treated_data_loader_val = torch.utils.data.DataLoader(treated_set_val,
                                                              batch_size=treated_batch_size,
                                                              shuffle=shuffle)
        control_data_loader_val = torch.utils.data.DataLoader(control_set_val,
                                                              batch_size=control_batch_size,
                                                              shuffle=shuffle)

        optimizer_shared = optim.Adam(self.dcn_shared.parameters(), lr=lr)
        optimizer_y1 = optim.Adam(self.dcn_y1.parameters(), lr=lr)
        optimizer_y0 = optim.Adam(self.dcn_y0.parameters(), lr=lr)

        lossF = nn.CrossEntropyLoss()
        min_loss = 100000.0
        dataset_loss = 0.0
        dataset_loss_train = 0.0
        dataset_loss_val = 0.0
        train_loss = 0
        val_loss = 0
        train_losses = []
        valid_losses = []
        avg_train_losses = []
        avg_valid_losses = []
        early_stopping = EarlyStopping_DCN(patience=80, verbose=True,
                                           model_shared_path="DCN_shared_checkpoint.pt",
                                           model_y1_path="DCN_y1_checkpoint.pt",
                                           model_y0_path="DCN_y0_checkpoint.pt")

        total_loss_train = 0
        total_loss_val = 0

        for epoch in range(epochs):
            epoch += 1
            self.dcn_shared.train()
            self.dcn_y1.train()
            self.dcn_y0.train()
            train_set_size = 0

            if epoch % 2 == 0:
                dataset_loss = 0
                # train treated
                for batch in treated_data_loader_train:
                    covariates_X, ps_score, y_f = batch

                    covariates_X = covariates_X.to(device)
                    ps_score = ps_score.squeeze().to(device)
                    train_set_size += covariates_X.size(0)
                    y1_hat = self.dcn_y1(self.dcn_shared(covariates_X, ps_score), ps_score)
                    y_f = y_f.long()
                    if torch.cuda.is_available():
                        loss = lossF(y1_hat.cuda(), y_f.cuda()).to(device)
                    else:
                        loss = lossF(y1_hat, y_f).to(device)

                    optimizer_shared.zero_grad()
                    optimizer_y1.zero_grad()
                    loss.backward()
                    optimizer_shared.step()
                    optimizer_y1.step()
                    total_loss_train += loss.item()

                train_losses.append(total_loss_train)
                dataset_loss = total_loss_train
                total_loss_train = 0

            elif epoch % 2 == 1:
                # train control

                for batch in control_data_loader_train:
                    covariates_X, ps_score, y_f = batch

                    covariates_X = covariates_X.to(device)
                    ps_score = ps_score.squeeze().to(device)

                    train_set_size += covariates_X.size(0)
                    y0_hat = self.dcn_y0(self.dcn_shared(covariates_X, ps_score), ps_score)
                    y_f = y_f.long()
                    if torch.cuda.is_available():
                        loss = F.cross_entropy(y0_hat.cuda(), y_f.cuda()).to(device)
                    else:
                        loss = F.cross_entropy(y0_hat, y_f).to(device)
                    optimizer_shared.zero_grad()
                    optimizer_y0.zero_grad()
                    loss.backward()
                    optimizer_shared.step()
                    optimizer_y0.step()
                    total_loss_train += loss.item()
                dataset_loss += total_loss_train

            ######################
            # validate the model #
            ######################
            # prep model for evaluation
            self.dcn_shared.eval()
            self.dcn_y1.eval()
            self.dcn_y0.eval()
            if epoch % 2 == 0:
                dataset_loss_val += 0
                # val treated
                for batch in treated_data_loader_val:
                    covariates_X, ps_score, y_f = batch
                    covariates_X = covariates_X.to(device)
                    ps_score = ps_score.squeeze().to(device)
                    y1_hat = self.dcn_y1(self.dcn_shared(covariates_X, ps_score), ps_score)
                    y_f = y_f.long()
                    if torch.cuda.is_available():
                        loss = lossF(y1_hat.cuda(), y_f.cuda()).to(device)
                    else:
                        loss = lossF(y1_hat, y_f).to(device)
                    total_loss_val += loss.item()
                valid_losses.append(total_loss_val)
                dataset_loss_val += total_loss_val
                total_loss_val = 0

            elif epoch % 2 == 1:
                # val control
                for batch in control_data_loader_val:
                    covariates_X, ps_score, y_f = batch
                    covariates_X = covariates_X.to(device)
                    ps_score = ps_score.squeeze().to(device)

                    train_set_size += covariates_X.size(0)
                    y0_hat = self.dcn_y0(self.dcn_shared(covariates_X, ps_score), ps_score)
                    y_f = y_f.long()
                    if torch.cuda.is_available():
                        loss = lossF(y0_hat, y_f).to(device)
                    else:
                        loss = lossF(y0_hat, y_f).to(device)
                    total_loss_val += loss.item()
                    dataset_loss_val += total_loss_val

            if epoch % 2 == 0:
                train_loss = np.average(np.array(train_losses))
                valid_loss = np.average(np.array(valid_losses))
                avg_train_losses.append(train_loss)
                avg_valid_losses.append(valid_loss)

                train_losses = []
                valid_losses = []
                early_stopping(valid_loss, self.dcn_shared, self.dcn_y1, self.dcn_y0)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            if epoch % 100 == 0:
                print("---->>>[[epoch: {0}/400]], Treated + Control loss, train: {1}, val: {2}".format(epoch,
                                                                                                       train_loss,
                                                                                                       valid_loss))

        self.dcn_shared.load_state_dict(torch.load("DCN_shared_checkpoint.pt"))
        self.dcn_y1.load_state_dict(torch.load("DCN_y1_checkpoint.pt"))
        self.dcn_y0.load_state_dict(torch.load("DCN_y0_checkpoint.pt"))

        if not ss:
            print(self.model_shared_path)
            print(self.model_y1_path)
            print(self.model_y0_path)

            torch.save(self.dcn_shared.state_dict(), self.model_shared_path)
            torch.save(self.dcn_y1.state_dict(), self.model_y1_path)
            torch.save(self.dcn_y0.state_dict(), self.model_y0_path)

    def eval(self, eval_parameters, device):
        treated_set = eval_parameters["treated_set"]
        control_set = eval_parameters["control_set"]

        print(self.model_shared_path)
        print(self.model_y1_path)
        print(self.model_y0_path)

        self.dcn_shared.load_state_dict(torch.load(self.model_shared_path,
                                                   map_location=device))
        self.dcn_y1.load_state_dict(torch.load(self.model_y1_path,
                                               map_location=device))
        self.dcn_y0.load_state_dict(torch.load(self.model_y0_path,
                                               map_location=device))

        self.dcn_shared.eval()
        self.dcn_y1.eval()
        self.dcn_y0.eval()
        treated_data_loader = torch.utils.data.DataLoader(treated_set,
                                                          shuffle=False)
        control_data_loader = torch.utils.data.DataLoader(control_set,
                                                          shuffle=False)

        predicted_ITE_list = []
        ITE_dict_list = []

        y_f_list = []
        y1_hat_list = []
        y0_hat_list = []
        e_list = []
        T_list = []

        for batch in treated_data_loader:
            covariates_X, ps_score, y_f, t, e = batch
            covariates_X = covariates_X.to(device)
            ps_score = ps_score.squeeze().to(device)
            pred_y1_hat = self.dcn_y1(self.dcn_shared(covariates_X, ps_score), ps_score)
            pred_y0_hat = self.dcn_y0(self.dcn_shared(covariates_X, ps_score), ps_score)
            _, y1_hat = torch.max(pred_y1_hat.data, 1)
            _, y0_hat = torch.max(pred_y0_hat.data, 1)

            predicted_ITE = y1_hat - y0_hat
            ITE_dict_list.append(self.create_ITE_Dict(covariates_X,
                                                      ps_score.item(), y_f.item(),
                                                      y1_hat.item(),
                                                      y0_hat.item(),
                                                      predicted_ITE.item()))
            y_f_list.append(y_f.item())
            y1_hat_list.append(y1_hat.item())
            y0_hat_list.append(y0_hat.item())
            e_list.append(e.item())
            T_list.append(t)
            predicted_ITE_list.append(predicted_ITE.item())

        for batch in control_data_loader:
            covariates_X, ps_score, y_f, t, e = batch
            covariates_X = covariates_X.to(device)
            ps_score = ps_score.squeeze().to(device)
            pred_y1_hat = self.dcn_y1(self.dcn_shared(covariates_X, ps_score), ps_score)
            pred_y0_hat = self.dcn_y0(self.dcn_shared(covariates_X, ps_score), ps_score)

            _, y1_hat = torch.max(pred_y1_hat.data, 1)
            _, y0_hat = torch.max(pred_y0_hat.data, 1)

            predicted_ITE = y1_hat - y0_hat

            ITE_dict_list.append(self.create_ITE_Dict(covariates_X,
                                                      ps_score.item(), y_f.item(),
                                                      y1_hat.item(),
                                                      y0_hat.item(),
                                                      predicted_ITE.item()))

            y_f_list.append(y_f.item())
            y1_hat_list.append(y1_hat.item())
            y0_hat_list.append(y0_hat.item())
            predicted_ITE_list.append(predicted_ITE.item())
            e_list.append(e.item())
            T_list.append(t)

        # print(err_treated_list)
        # print(err_control_list)
        return {
            "predicted_ITE": predicted_ITE_list,
            "ITE_dict_list": ITE_dict_list,
            "y1_hat_list": y1_hat_list,
            "y0_hat_list": y0_hat_list,
            "e_list": e_list,
            "yf_list": y_f_list,
            "T_list": T_list
        }

    def eval_semi_supervised(self, eval_parameters, device, treated_flag):
        eval_set = eval_parameters["eval_set"]

        self.dcn_shared.eval()
        self.dcn_y1.eval()
        self.dcn_y0.eval()
        treated_data_loader = torch.utils.data.DataLoader(eval_set,
                                                          shuffle=False)

        y_f_list = []
        y_1_list = []
        y_0_list = []

        for batch in treated_data_loader:
            covariates_X, ps_score = batch
            covariates_X = covariates_X.to(device)
            ps_score = ps_score.squeeze().to(device)
            y1_hat = self.dcn_y1(self.dcn_shared(covariates_X, ps_score), ps_score)
            y0_hat = self.dcn_y0(self.dcn_shared(covariates_X, ps_score), ps_score)
            if treated_flag:
                _, predicted_y1 = torch.max(y1_hat.data, 1)
                y_f_list.append(predicted_y1.item())
            else:
                _, predicted_y0 = torch.max(y0_hat.data, 1)
                y_f_list.append(y0_hat.item())

            y_1_list.append(0)
            y_0_list.append(0)

        return {
            "y_f_list": np.array(y_f_list),
            "y_0_list": np.array(y_1_list),
            "y_1_list": np.array(y_0_list)
        }

    @staticmethod
    def create_ITE_Dict(covariates_X, ps_score, y_f,
                        y1_hat,
                        y0_hat,
                        predicted_ITE):
        result_dict = OrderedDict()
        covariate_list = [element.item() for element in covariates_X.flatten()]
        idx = 0
        for item in covariate_list:
            idx += 1
            result_dict["X" + str(idx)] = item

        result_dict["ps_score"] = ps_score
        result_dict["factual"] = y_f
        result_dict["y1_hat"] = y1_hat
        result_dict["y0_hat"] = y0_hat
        result_dict["predicted_ITE"] = predicted_ITE

        return result_dict
