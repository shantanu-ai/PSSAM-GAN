import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from Constants import Constants
from Utils import Utils


# training_flag = "eval" / "train_constant_dropout" / "train_PD"

class DCN(nn.Module):
    def __init__(self, training_mode, input_nodes):
        super(DCN, self).__init__()
        self.training = training_mode

        # shared layer
        self.shared1 = nn.Linear(in_features=input_nodes, out_features=200)
        nn.init.xavier_uniform_(self.shared1.weight)

        self.shared2 = nn.Linear(in_features=200, out_features=200)
        nn.init.xavier_uniform_(self.shared2.weight)

        # potential outcome1 Y(1)
        self.hidden1_Y1 = nn.Linear(in_features=200, out_features=200)
        nn.init.xavier_uniform_(self.hidden1_Y1.weight)

        self.hidden2_Y1 = nn.Linear(in_features=200, out_features=200)
        nn.init.xavier_uniform_(self.hidden2_Y1.weight)

        self.out_Y1 = nn.Linear(in_features=200, out_features=1)
        nn.init.xavier_uniform_(self.out_Y1.weight)

        # potential outcome1 Y(0)
        self.hidden1_Y0 = nn.Linear(in_features=200, out_features=200)
        nn.init.xavier_uniform_(self.hidden1_Y0.weight)

        self.hidden2_Y0 = nn.Linear(in_features=200, out_features=200)
        nn.init.xavier_uniform_(self.hidden2_Y0.weight)

        self.out_Y0 = nn.Linear(in_features=200, out_features=1)
        nn.init.xavier_uniform_(self.out_Y0.weight)

        self.dropout_2 = nn.Dropout(p=0.2)
        self.dropout_5 = nn.Dropout(p=0.5)

    def forward(self, x, ps_score):
        if torch.cuda.is_available():
            x = x.float().cuda()
        else:
            x = x.float()

        if self.training == Constants.DCN_EVALUATION:
            y1, y0 = self.__eval_net(x)
        elif self.training == Constants.DCN_TRAIN_PD:
            y1, y0 = self.__train_net_PD(x, ps_score)
        elif self.training == Constants.DCN_TRAIN_CONSTANT_DROPOUT:
            y1, y0 = self.__train_net_constant_dropout(x, ps_score)
        elif self.training == Constants.DCN_TRAIN_NO_DROPOUT:
            y1, y0 = self.__train_net_no_droput(x)

        return y1, y0

    def __train_net_no_droput(self, x):
        # shared layers
        x = F.relu(self.shared1(x))
        x = F.relu(self.shared2(x))

        # potential outcome1 Y(1)
        y1 = F.relu(self.hidden1_Y1(x))
        y1 = F.relu(self.hidden2_Y1(y1))
        y1 = self.out_Y1(y1)

        # potential outcome1 Y(0)
        y0 = F.relu(self.hidden1_Y0(x))
        y0 = F.relu(self.hidden2_Y0(y0))
        y0 = self.out_Y0(y0)

        return y1, y0

    def __train_net_constant_dropout(self, x, ps_score):
        if ps_score == 0.2:
            drop_out = self.dropout_2
        elif ps_score == 0.5:
            drop_out = self.dropout_5

        # shared layers
        x = F.relu(drop_out(self.shared1(x)))
        x = F.relu(drop_out(self.shared2(x)))

        # potential outcome1 Y(1)
        y1 = F.relu(drop_out(self.hidden1_Y1(x)))
        y1 = F.relu(drop_out(self.hidden2_Y1(y1)))
        y1 = self.out_Y1(y1)

        # potential outcome1 Y(0)
        y0 = F.relu(drop_out(self.hidden1_Y0(x)))
        y0 = F.relu(drop_out(self.hidden2_Y0(y0)))
        y0 = self.out_Y0(y0)

        return y1, y0

    def __train_net_PD(self, x, ps_score):
        entropy = Utils.get_shanon_entropy(ps_score.item())
        dropout_prob = Utils.get_dropout_probability(entropy, gama=1)

        # shared layers
        shared_mask = Utils.get_dropout_mask(dropout_prob, self.shared1(x))
        x = F.relu(shared_mask * self.shared1(x))
        x = F.relu(shared_mask * self.shared2(x))

        # potential outcome1 Y(1)
        y1_mask = Utils.get_dropout_mask(dropout_prob, self.hidden1_Y1(x))
        y1 = F.relu(y1_mask * self.hidden1_Y1(x))
        y1 = F.relu(y1_mask * self.hidden2_Y1(y1))
        y1 = self.out_Y1(y1)

        # potential outcome1 Y(0)
        y0_mask = Utils.get_dropout_mask(dropout_prob, self.hidden1_Y0(x))
        y0 = F.relu(y0_mask * self.hidden1_Y0(x))
        y0 = F.relu(y0_mask * self.hidden2_Y0(y0))
        y0 = self.out_Y0(y0)

        return y1, y0

    def __eval_net(self, x):
        # shared layers
        x = F.relu(self.shared1(x))
        x = F.relu(self.shared2(x))

        # potential outcome1 Y(1)
        y1 = F.relu(self.hidden1_Y1(x))
        y1 = F.relu(self.hidden2_Y1(y1))
        y1 = self.out_Y1(y1)

        # potential outcome1 Y(0)
        y0 = F.relu(self.hidden1_Y0(x))
        y0 = F.relu(self.hidden2_Y0(y0))
        y0 = self.out_Y0(y0)

        return y1, y0
