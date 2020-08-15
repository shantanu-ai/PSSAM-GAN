import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Utils import Utils
from Constants import Constants
from Propensity_socre_network import Propensity_socre_network
from Utils import Utils
from PS_Manager import PS_Manager
from PS_Treated_Generator import PS_Treated_Generator

from GAN import Generator, Discriminator
from sklearn.neighbors import NearestNeighbors

from GAN_Manager import GAN_Manager
from Utils import Utils

from matplotlib import pyplot
from torch.autograd.variable import Variable
from collections import OrderedDict
from scipy.special import expit

import os
from os.path import join
import sys

train_path = "Dataset/jobs_DW_bin.train.npz"
test_path = "Dataset/jobs_DW_bin.test.npz"

this_directory = os.path.dirname(os.path.realpath(__file__))
train_file_path = join(this_directory, train_path)
test_file_path = join(this_directory, test_path)
train_set = np.load(train_file_path)
test_set = np.load(test_file_path)

print(test_set.size)
