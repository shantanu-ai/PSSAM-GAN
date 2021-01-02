import torch
import torch.nn.functional as F
import torch.optim as optim

from Propensity_net_NN import Propensity_net_NN
from Utils import Utils


class Propensity_socre_network:
    def __init__(self, input_nodes, device):
        self.network = Propensity_net_NN(input_nodes).to(device)
        self.phase = None

    def set_train_mode(self, phase):
        self.phase = phase

    def get_ps_model(self):
        return self.network

    def train(self, train_parameters, device):
        print(".. PS Training started ..")
        epochs = train_parameters["epochs"]
        batch_size = train_parameters["batch_size"]
        lr = train_parameters["lr"]
        shuffle = train_parameters["shuffle"]
        train_set = train_parameters["train_set"]

        data_loader_train = torch.utils.data.DataLoader(train_set,
                                                        batch_size=batch_size,
                                                        shuffle=shuffle)
        self.network.set_train_mode(self.phase)

        optimizer = optim.Adam(self.network.parameters(), lr=lr)
        for epoch in range(epochs):
            epoch += 1
            self.network.train()  # Set model to training mode
            total_loss = 0
            total_correct = 0
            train_set_size = 0

            for batch in data_loader_train:
                covariates, treatment = batch
                covariates = covariates.to(device)
                treatment = treatment.squeeze().to(device)

                covariates = covariates[:, :-2]
                train_set_size += covariates.size(0)

                treatment_pred = self.network(covariates)

                loss = F.cross_entropy(treatment_pred, treatment).to(device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_correct += Utils.get_num_correct(treatment_pred, treatment)

            pred_accuracy = total_correct / train_set_size
            if epoch % 25 == 0:
                print("Epoch: {0}, loss: {1}, correct: {2}/{3}, accuracy: {4}".
                      format(epoch, total_loss, total_correct, train_set_size, pred_accuracy))
        print("Training Completed..")

    def eval(self, eval_parameters, device, eval_from_GAN=False):
        eval_set = eval_parameters["eval_set"]
        self.network.set_train_mode(self.phase)
        self.network.eval()
        data_loader = torch.utils.data.DataLoader(eval_set, shuffle=False)
        eval_set_size = 0
        prop_score_list = []
        for batch in data_loader:
            covariates, treatment = batch

            covariates = covariates.to(device)
            if not eval_from_GAN:
                covariates = covariates[:, :-2]

            eval_set_size += covariates.size(0)

            treatment_pred = self.network(covariates)
            treatment_pred = treatment_pred.squeeze()
            prop_score_list.append(treatment_pred[1].item())

        return prop_score_list
