import os
from os.path import join

import numpy as np

from Utils import Utils

train_path = "Dataset/jobs_DW_bin.new.10.train.npz"
test_path = "Dataset/jobs_DW_bin.new.10.test.npz"

this_directory = os.path.dirname(os.path.realpath(__file__))
train_file_path = join(this_directory, train_path)
test_file_path = join(this_directory, test_path)
train_arr = np.load(train_file_path)
test_arr = np.load(test_file_path)
iter_id = 1

np_train_X = train_arr['x'][:, :, iter_id]
np_train_T = Utils.convert_to_col_vector(train_arr['t'][:, iter_id])
np_train_e = Utils.convert_to_col_vector(train_arr['e'][:, iter_id])
np_train_yf = Utils.convert_to_col_vector(train_arr['yf'][:, iter_id])
print(np_train_X.shape)
print(np_train_T.shape)

train_X = np.concatenate((np_train_X, np_train_T, np_train_e, np_train_yf), axis=1)
# train_X, val_X, train_T, val_T = \
#     Utils.test_train_split(train_X, np_train_T, split_size=0.90)

np_test_X = test_arr['x'][:, :, iter_id]
np_test_T = Utils.convert_to_col_vector(test_arr['t'][:, iter_id])
np_test_e = Utils.convert_to_col_vector(test_arr['e'][:, iter_id])
np_test_yf = Utils.convert_to_col_vector(test_arr['yf'][:, iter_id])

test_X = np.concatenate((np_test_X, np_test_T, np_test_e, np_test_yf), axis=1)

print("Numpy Train Statistics:")
print(train_X.shape)
# print(train_T.shape)

print("Numpy Val Statistics:")
# print(val_X.shape)
# print(val_T.shape)

print(" Numpy Test Statistics:")
print(test_X.shape)
print(np_test_T.shape)

np.savetxt(join(this_directory, 'train.csv'), train_X, delimiter=",")
np.savetxt(join(this_directory, 'test.csv'), test_X, delimiter=",")
# my_data = np.genfromtxt(this_directory + '/my_file.csv', delimiter=',')

# for key, value in data.items():
#     np.savetxt("somepath" + key + ".csv", value)
