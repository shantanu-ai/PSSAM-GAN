from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from Constants import Constants
from DCN import DCN


class DCN_network:
    def train(self, train_parameters, device, train_mode=Constants.DCN_TRAIN_PD):
        epochs = train_parameters["epochs"]
        treated_batch_size = train_parameters["treated_batch_size"]
        control_batch_size = train_parameters["control_batch_size"]
        lr = train_parameters["lr"]
        shuffle = train_parameters["shuffle"]
        model_save_path = train_parameters["model_save_path"].format(epochs, lr)
        treated_set_train = train_parameters["treated_set_train"]
        control_set_train = train_parameters["control_set_train"]

        input_nodes = train_parameters["input_nodes"]

        print("Saved model path: {0}".format(model_save_path))

        treated_data_loader_train = torch.utils.data.DataLoader(treated_set_train,
                                                                batch_size=treated_batch_size,
                                                                shuffle=shuffle,
                                                                num_workers=1)

        control_data_loader_train = torch.utils.data.DataLoader(control_set_train,
                                                                batch_size=control_batch_size,
                                                                shuffle=shuffle,
                                                                num_workers=1)

        network = DCN(training_mode=train_mode,
                      input_nodes=input_nodes).to(device)
        optimizer = optim.Adam(network.parameters(), lr=lr)
        lossF = nn.MSELoss()
        min_loss = 100000.0
        dataset_loss = 0.0
        print(".. Training started ..")
        print(device)
        for epoch in range(epochs):
            network.train()
            total_loss = 0
            train_set_size = 0

            if epoch % 2 == 0:
                dataset_loss = 0
                # train treated
                network.hidden1_Y1.weight.requires_grad = True
                network.hidden1_Y1.bias.requires_grad = True
                network.hidden2_Y1.weight.requires_grad = True
                network.hidden2_Y1.bias.requires_grad = True
                network.out_Y1.weight.requires_grad = True
                network.out_Y1.bias.requires_grad = True

                network.hidden1_Y0.weight.requires_grad = False
                network.hidden1_Y0.bias.requires_grad = False
                network.hidden2_Y0.weight.requires_grad = False
                network.hidden2_Y0.bias.requires_grad = False
                network.out_Y0.weight.requires_grad = False
                network.out_Y0.bias.requires_grad = False

                for batch in treated_data_loader_train:
                    covariates_X, ps_score, y_f, y_cf = batch
                    covariates_X = covariates_X.to(device)
                    ps_score = ps_score.squeeze().to(device)
                    train_set_size += covariates_X.size(0)
                    treatment_pred = network(covariates_X, ps_score)
                    # treatment_pred[0] -> y1
                    # treatment_pred[1] -> y0
                    yf_predicted = treatment_pred[0]
                    if torch.cuda.is_available():
                        loss = lossF(yf_predicted.float().cuda(),
                                     y_f.float().cuda()).to(device)
                    else:
                        loss = lossF(yf_predicted.float(),
                                     y_f.float()).to(device)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                dataset_loss = total_loss

            elif epoch % 2 == 1:
                # train controlled
                network.hidden1_Y1.weight.requires_grad = False
                network.hidden1_Y1.bias.requires_grad = False
                network.hidden2_Y1.weight.requires_grad = False
                network.hidden2_Y1.bias.requires_grad = False
                network.out_Y1.weight.requires_grad = False
                network.out_Y1.bias.requires_grad = False

                network.hidden1_Y0.weight.requires_grad = True
                network.hidden1_Y0.bias.requires_grad = True
                network.hidden2_Y0.weight.requires_grad = True
                network.hidden2_Y0.bias.requires_grad = True
                network.out_Y0.weight.requires_grad = True
                network.out_Y0.bias.requires_grad = True

                for batch in control_data_loader_train:
                    covariates_X, ps_score, y_f, y_cf = batch
                    covariates_X = covariates_X.to(device)
                    ps_score = ps_score.squeeze().to(device)

                    train_set_size += covariates_X.size(0)
                    treatment_pred = network(covariates_X, ps_score)
                    # treatment_pred[0] -> y1
                    # treatment_pred[1] -> y0
                    yf_predicted = treatment_pred[1]
                    if torch.cuda.is_available():
                        loss = lossF(yf_predicted.float().cuda(),
                                     y_f.float().cuda()).to(device)
                    else:
                        loss = lossF(yf_predicted.float(),
                                     y_f.float()).to(device)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                dataset_loss = dataset_loss + total_loss

            if epoch % 10 == 9:
                print("epoch: {0}, Treated + Control loss: {1}".format(epoch, dataset_loss))
        torch.save(network.state_dict(), model_save_path)

    def eval(self, eval_parameters, device, input_nodes, train_mode):
        treated_set = eval_parameters["treated_set"]
        control_set = eval_parameters["control_set"]
        model_path = eval_parameters["model_save_path"]
        network = DCN(training_mode=train_mode, input_nodes=input_nodes).to(device)
        network.load_state_dict(torch.load(model_path, map_location=device))
        network.eval()
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
            treatment_pred = network(covariates_X, ps_score)

            predicted_ITE = treatment_pred[0] - treatment_pred[1]
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
            treatment_pred = network(covariates_X, ps_score)

            predicted_ITE = treatment_pred[0] - treatment_pred[1]
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

    @staticmethod
    def eval_semi_supervised(eval_parameters, device, input_nodes, train_mode, treated_flag):
        eval_set = eval_parameters["eval_set"]
        model_path = eval_parameters["model_save_path"]
        network = DCN(training_mode=train_mode, input_nodes=input_nodes).to(device)
        network.load_state_dict(torch.load(model_path, map_location=device))
        network.eval()
        treated_data_loader = torch.utils.data.DataLoader(eval_set,
                                                          shuffle=False, num_workers=1)

        y_f_list = []
        y_cf_list = []

        for batch in treated_data_loader:
            covariates_X, ps_score = batch
            covariates_X = covariates_X.to(device)
            ps_score = ps_score.squeeze().to(device)
            treatment_pred = network(covariates_X, ps_score)
            if treated_flag:
                y_f_list.append(treatment_pred[0].item())
                y_cf_list.append(treatment_pred[1].item())
            else:
                y_f_list.append(treatment_pred[1].item())
                y_cf_list.append(treatment_pred[0].item())

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
