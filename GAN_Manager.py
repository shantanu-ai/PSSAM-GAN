import numpy as np
import torch
import torch.optim as optim

from GAN import Generator, Discriminator
from Propensity_socre_network import Propensity_socre_network
from Utils import Utils


class GAN_Manager:
    def train_GAN(self, train_parameters, device):
        epochs = train_parameters["epochs"]
        train_set = train_parameters["train_set"]
        lr = train_parameters["lr"]
        shuffle = train_parameters["shuffle"]
        discriminator_in_nodes = train_parameters["discriminator_in_nodes"]
        generator_out_nodes = train_parameters["generator_out_nodes"]
        batch_size = train_parameters["batch_size"]
        prop_score_NN_model_path = train_parameters["prop_score_NN_model_path"]
        BETA = train_parameters["BETA"]

        data_loader_train = torch.utils.data.DataLoader(train_set,
                                                        batch_size=batch_size,
                                                        shuffle=shuffle,
                                                        num_workers=1)

        generator = Generator(out_nodes=generator_out_nodes).to(device)
        discriminator = Discriminator(in_nodes=discriminator_in_nodes).to(device)

        loss_func = torch.nn.BCELoss()
        optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.4, 0.999))
        optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.4, 0.999))

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        print("GAN Training started..")

        for epoch in range(epochs):
            epoch += 1
            generator.train()
            discriminator.train()
            total_G_loss = 0
            total_D_loss = 0
            total_prop_loss = 0
            total_size = 0

            for batch in data_loader_train:
                covariates_X, ps_score, y_f, y_cf = batch
                covariates_X = covariates_X.to(device)
                covariates_X_size = covariates_X.size(0)
                total_size += covariates_X_size
                ps_score = ps_score.squeeze().to(device)

                # Labels - Real, Fake
                valid_labels = Tensor(covariates_X_size, 1).fill_(1.0)
                fake_labels = Tensor(covariates_X_size, 1).fill_(0.0)

                optimizer_G.zero_grad()
                gen_input = Tensor(np.random.normal(0, 1, (covariates_X_size, 25)))
                gen_treated = generator(gen_input)

                # train generator
                output_g = discriminator(gen_treated)
                generator_loss = loss_func(output_g, valid_labels)
                prop_loss = self.__cal_propensity_loss(ps_score, prop_score_NN_model_path,
                                                       gen_treated, device)
                g_loss = generator_loss + (BETA * prop_loss)
                # g_loss = generator_loss
                g_loss.backward()
                optimizer_G.step()
                total_G_loss += g_loss.item()
                total_prop_loss += prop_loss.item()
                # total_prop_loss += 0

                # train discriminator
                optimizer_D.zero_grad()
                output_d_real = discriminator(covariates_X)
                output_d_fake = discriminator(gen_treated.detach().cpu())
                real_loss = loss_func(output_d_real, valid_labels)
                fake_loss = loss_func(output_d_fake, fake_labels)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_D.step()
                total_D_loss += d_loss.item()

            print("Epoch: {0}, G_loss: {1}, D_loss: {2}, Prop_loss: {3}, "
                  .format(epoch, total_G_loss, total_D_loss, total_prop_loss))

        return generator

    def eval_GAN(self, eval_size, generator, prop_score_NN_model_path, device):
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        gen_input = Tensor(np.random.normal(0, 1, (eval_size, 25)))
        generator.eval()
        gen_treated = generator(gen_input)
        ps_score_list_treated = self.__get_propensity_score(gen_treated, prop_score_NN_model_path, device)
        return gen_treated, ps_score_list_treated

    def __cal_propensity_loss(self, ps_score_control, prop_score_NN_model_path, gen_treated, device):
        ps_score_list_treated = self.__get_propensity_score(gen_treated, prop_score_NN_model_path, device)

        Tensor = torch.cuda.DoubleTensor if torch.cuda.is_available() else torch.DoubleTensor
        ps_score_treated = Tensor(ps_score_list_treated)

        prop_loss = torch.sum((torch.sub(ps_score_treated, ps_score_control)) ** 2)
        return prop_loss

    @staticmethod
    def __get_propensity_score(gen_treated, prop_score_NN_model_path, device):
        Y = np.ones(gen_treated.size(0))
        eval_set = Utils.convert_to_tensor(gen_treated.detach().cpu().numpy(), Y)

        eval_parameters_ps_net = {
            "eval_set": eval_set,
            "model_path": prop_score_NN_model_path,
            "input_nodes": 25
        }
        ps_net_NN = Propensity_socre_network()
        ps_score_list_treated = ps_net_NN.eval(eval_parameters_ps_net, device,
                                               phase="eval", eval_from_GAN=True)

        return ps_score_list_treated
