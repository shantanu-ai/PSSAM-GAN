import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.out_Y1 = nn.Linear(in_features=outcome_nodes, out_features=2)
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

        self.out_Y0 = nn.Linear(in_features=outcome_nodes, out_features=2)
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
