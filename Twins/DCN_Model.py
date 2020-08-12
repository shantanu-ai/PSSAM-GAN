import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

from Constants import Constants
from Utils import Utils


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

        self.out_Y1 = nn.Linear(in_features=200, out_features=2)
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

        self.out_Y0 = nn.Linear(in_features=200, out_features=2)
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
