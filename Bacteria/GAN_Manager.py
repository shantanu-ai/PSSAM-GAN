import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import Variable

from Constants import Constants
from GAN import Generator, Discriminator
from Utils import Utils


class GAN_Manager:
    def __init__(self, discriminator_in_nodes, generator_out_nodes, ps_model,
                 ps_model_type, device):
        self.discriminator = Discriminator(in_nodes=discriminator_in_nodes).to(device)
        self.discriminator.apply(self.__weights_init)

        self.generator = Generator(out_nodes=generator_out_nodes).to(device)
        self.generator.apply(self.__weights_init)

        self.loss = nn.BCELoss()
        self.ps_model = ps_model
        self.ps_model_type = ps_model_type

    def get_generator(self):
        return self.generator

    def train_GAN(self, train_parameters, device):
        epochs = train_parameters["epochs"]
        train_set = train_parameters["train_set"]
        lr = train_parameters["lr"]
        shuffle = train_parameters["shuffle"]
        batch_size = train_parameters["batch_size"]
        BETA = train_parameters["BETA"]

        data_loader_train = torch.utils.data.DataLoader(train_set,
                                                        batch_size=batch_size,
                                                        shuffle=shuffle)

        g_optimizer = optim.Adam(self.generator.parameters(), lr=lr)
        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr)

        for epoch in range(epochs):
            epoch += 1

            total_G_loss = 0
            total_D_loss = 0
            total_prop_loss = 0
            total_d_pred_real = 0
            total_d_pred_fake = 0

            for batch in data_loader_train:
                covariates_X_control, ps_score_control = batch
                covariates_X_control = covariates_X_control.to(device)
                covariates_X_control_size = covariates_X_control.size(0)
                ps_score_control = ps_score_control.squeeze().to(device)

                # 1. Train Discriminator
                real_data = covariates_X_control

                # Generate fake data
                fake_data = self.generator(self.__noise(covariates_X_control_size)).detach()
                # Train D
                d_error, d_pred_real, d_pred_fake = self.__train_discriminator(d_optimizer,
                                                                               real_data, fake_data)
                total_D_loss += d_error
                total_d_pred_real += d_pred_real
                total_d_pred_fake += d_pred_fake

                # 2. Train Generator
                # Generate fake data
                fake_data = self.generator(self.__noise(covariates_X_control_size))
                # Train G
                error_g, prop_loss = self.__train_generator(g_optimizer, fake_data, BETA, ps_score_control,
                                                            device)
                total_G_loss += error_g
                total_prop_loss += prop_loss

            if epoch % 1000 == 0:
                print("Epoch: {0}, D_loss: {1}, D_score_real: {2}, D_score_Fake: {3}, G_loss: {4}, "
                      "Prop_loss: {5}"
                      .format(epoch,
                              total_D_loss, total_d_pred_real, total_d_pred_fake, total_G_loss, total_prop_loss))

    def eval_GAN(self, eval_size, device):
        treated_g = self.generator(self.__noise(eval_size))
        ps_score_list_treated = self.__get_propensity_score(treated_g, device)
        return treated_g, ps_score_list_treated

    def __cal_propensity_loss(self, ps_score_control,
                              gen_treated, device):
        ps_score_list_treated = self.__get_propensity_score(gen_treated, device)

        ps_score_treated = torch.tensor(ps_score_list_treated).to(device)
        ps_score_control = ps_score_control.to(device)
        prop_loss = torch.sum((torch.sub(ps_score_treated.float(),
                                         ps_score_control.float())) ** 2)
        return prop_loss

    def __get_propensity_score(self, gen_treated, device):
        if self.ps_model_type == Constants.PS_MODEL_NN:
            return self.__get_propensity_score_NN(gen_treated, device)
        else:
            return self.__get_propensity_score_LR(gen_treated)

    def __get_propensity_score_LR(self, gen_treated):
        ps_score_list_treated = self.ps_model.predict_proba(
            gen_treated.cpu().detach().numpy())[:, -1].tolist()
        return ps_score_list_treated

    def __get_propensity_score_NN(self, gen_treated, device):
        # Assign Treated
        Y = np.ones(gen_treated.size(0))
        eval_set = Utils.convert_to_tensor(gen_treated.cpu().detach().numpy(), Y)
        ps_eval_parameters_NN = {
            "eval_set": eval_set
        }
        ps_score_list_treated = self.ps_model.eval(ps_eval_parameters_NN, device,
                                                   eval_from_GAN=True)
        return ps_score_list_treated

    @staticmethod
    def __noise(_size):
        n = Variable(torch.normal(mean=0, std=1, size=(_size, Constants.GAN_GENERATOR_IN_NODES)))
        # print(n.size())
        if torch.cuda.is_available(): return n.cuda()
        return n

    @staticmethod
    def __weights_init(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

    @staticmethod
    def __real_data_target(size):
        data = Variable(torch.ones(size, 1))
        if torch.cuda.is_available(): return data.cuda()
        return data

    @staticmethod
    def __fake_data_target(size):
        data = Variable(torch.zeros(size, 1))
        if torch.cuda.is_available(): return data.cuda()
        return data

    def __train_discriminator(self, optimizer, real_data, fake_data):
        # Reset gradients
        optimizer.zero_grad()

        # 1.1 Train on Real Data
        prediction_real = self.discriminator(real_data)
        real_score = torch.mean(prediction_real).item()

        # Calculate error and back propagate
        error_real = self.loss(prediction_real, self.__real_data_target(real_data.size(0)))
        error_real.backward()

        # 1.2 Train on Fake Data
        prediction_fake = self.discriminator(fake_data)
        fake_score = torch.mean(prediction_fake).item()
        # Calculate error and backpropagate
        error_fake = self.loss(prediction_fake, self.__fake_data_target(real_data.size(0)))
        error_fake.backward()

        # 1.3 Update weights with gradients
        optimizer.step()
        loss_D = error_real + error_fake
        # Return error
        return loss_D.item(), real_score, fake_score

    def __train_generator(self, optimizer, fake_data, BETA, ps_score_control,
                          device):
        # 2. Train Generator
        # Reset gradients
        optimizer.zero_grad()
        # Sample noise and generate fake data
        predicted_D = self.discriminator(fake_data)
        # Calculate error and back propagate
        ps_score_control = ps_score_control.to(device)
        fake_data = fake_data.to(device)
        error_g = self.loss(predicted_D, self.__real_data_target(predicted_D.size(0)))
        prop_loss = self.__cal_propensity_loss(ps_score_control,
                                               fake_data, device)
        error = error_g + (BETA * prop_loss)
        error.backward()
        # Update weights with gradients
        optimizer.step()
        # Return error
        return error_g.item(), prop_loss.item()
