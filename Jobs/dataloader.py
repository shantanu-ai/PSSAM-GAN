import os

import numpy as np
import pandas as pd
from scipy.special import expit

from Utils import Utils


class DataLoader:
    def preprocess_for_graphs(self, csv_path):
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), csv_path), header=None)
        return self.__convert_to_numpy(df)

    @staticmethod
    def load_train_test_jobs(train_path, test_path, iter_id):
        train_arr = np.load(train_path)
        test_arr = np.load(test_path)
        np_train_X = train_arr['x'][:, :, iter_id]
        np_train_T = Utils.convert_to_col_vector(train_arr['t'][:, iter_id])
        np_train_e = Utils.convert_to_col_vector(train_arr['e'][:, iter_id])
        np_train_yf = Utils.convert_to_col_vector(train_arr['yf'][:, iter_id])

        train_X = np.concatenate((np_train_X, np_train_e, np_train_yf), axis=1)
        train_X, val_X, train_T, val_T = \
            Utils.test_train_split(train_X, np_train_T, split_size=0.56)

        np_test_X = test_arr['x'][:, :, iter_id]
        np_test_T = Utils.convert_to_col_vector(test_arr['t'][:, iter_id])
        np_test_e = Utils.convert_to_col_vector(test_arr['e'][:, iter_id])
        np_test_yf = Utils.convert_to_col_vector(test_arr['yf'][:, iter_id])

        test_X = np.concatenate((np_test_X, np_test_e, np_test_yf), axis=1)

        print("Numpy Train Statistics:")
        print(train_X.shape)
        print(train_T.shape)

        print("Numpy Val Statistics:")
        print(val_X.shape)
        print(val_T.shape)

        print(" Numpy Test Statistics:")
        print(test_X.shape)
        print(np_test_T.shape)

        # X -> x1.. x17, e, yf -> (19, 1)
        return train_X, test_X, val_X, train_T, np_test_T, val_T

    def preprocess_data_from_csv_augmented(self, csv_path, split_size):
        # print(".. Data Loading synthetic..")
        # data load
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), csv_path), header=None)
        np_covariates_X, np_treatment_Y = self.__convert_to_numpy_augmented(df)
        # print("ps_np_covariates_X: {0}".format(np_covariates_X.shape))
        # print("ps_np_treatment_Y: {0}".format(np_treatment_Y.shape))

        np_covariates_X_train, np_covariates_X_test, np_covariates_Y_train, np_covariates_Y_test = \
            Utils.test_train_split(np_covariates_X, np_treatment_Y, split_size)
        # print("np_covariates_X_train: {0}".format(np_covariates_X_train.shape))
        # print("np_covariates_Y_train: {0}".format(np_covariates_Y_train.shape))
        # print("---" * 20)
        # print("np_covariates_X_test: {0}".format(np_covariates_X_test.shape))
        # print("np_covariates_Y_test: {0}".format(np_covariates_Y_test.shape))
        return np_covariates_X_train, np_covariates_X_test, np_covariates_Y_train, np_covariates_Y_test

    @staticmethod
    def convert_to_tensor(ps_np_covariates_X, ps_np_treatment_T):
        return Utils.convert_to_tensor(ps_np_covariates_X, ps_np_treatment_T)

    def prepare_tensor_for_DCN(self, ps_np_covariates_X, ps_np_treatment_T, ps_list,
                               is_synthetic):
        # ps_np_covariates_X -> X1 .. X17, y_f
        # print("ps_np_covariates_X: {0}".format(ps_np_covariates_X.shape))
        # print("ps_np_treatment_Y: {0}".format(ps_np_treatment_Y.shape))
        X = Utils.concat_np_arr(ps_np_covariates_X, ps_np_treatment_T, axis=1)

        # col of X -> x1 .. x17, Y_f, T, Ps
        X = Utils.concat_np_arr(X, np.array([ps_list]).T, axis=1)
        print("Big X: {0}".format(X.shape))

        df_X = pd.DataFrame(X)
        treated_df_X, treated_ps_score, treated_df_Y_f, treated_df_e = \
            self.__preprocess_data_for_DCN(df_X, treatment_index=1,
                                           is_synthetic=is_synthetic)

        control_df_X, control_ps_score, control_df_Y_f, control_df_e = \
            self.__preprocess_data_for_DCN(df_X, treatment_index=0,
                                           is_synthetic=is_synthetic)

        np_treated_df_X, np_treated_ps_score, np_treated_df_Y_f, np_treated_df_e = \
            self.__convert_to_numpy_DCN(treated_df_X, treated_ps_score, treated_df_Y_f, treated_df_e)

        np_control_df_X, np_control_ps_score, np_control_df_Y_f, np_control_df_e = \
            self.__convert_to_numpy_DCN(control_df_X, control_ps_score, control_df_Y_f, control_df_e)

        # np_treated_df_Y_f = Utils.convert_to_col_vector()

        print(" Treated Statistics ==>")
        print(np_treated_df_X.shape)
        print(" Control Statistics ==>")
        print(np_control_df_X.shape)

        return {
            "treated_data": (np_treated_df_X, np_treated_ps_score,
                             np_treated_df_Y_f, np_treated_df_e),
            "control_data": (np_control_df_X, np_control_ps_score,
                             np_control_df_Y_f, np_control_df_e)
        }

    @staticmethod
    def __convert_to_numpy(df):
        covariates_X = df.iloc[:, 5:]
        treatment_Y = df.iloc[:, 0:1]
        outcomes_Y = df.iloc[:, 1:3]

        np_covariates_X = Utils.convert_df_to_np_arr(covariates_X)
        np_outcomes_Y = Utils.convert_df_to_np_arr(outcomes_Y)
        np_X = Utils.concat_np_arr(np_covariates_X, np_outcomes_Y, axis=1)

        np_treatment_Y = Utils.convert_df_to_np_arr(treatment_Y)

        return np_X, np_treatment_Y

    @staticmethod
    def __convert_to_numpy_augmented(df):
        covariates_X = df.iloc[:, 5:]
        treatment_Y = df.iloc[:, 0:1]
        outcomes_Y = df.iloc[:, 1:3]

        np_covariates_X = Utils.convert_df_to_np_arr(covariates_X)
        np_std = np.std(np_covariates_X, axis=0)
        np_outcomes_Y = Utils.convert_df_to_np_arr(outcomes_Y)

        noise = np.empty([747, 25])
        id = -1
        for std in np_std:
            id += 1
            noise[:, id] = np.random.normal(0, 1.96 * std)

        random_correlated = np_covariates_X + noise

        random_X = np.random.permutation(np.random.random((747, 25)) * 10)
        np_covariates_X = np.concatenate((np_covariates_X, random_X), axis=1)
        np_covariates_X = np.concatenate((np_covariates_X, random_correlated), axis=1)
        np_X = Utils.concat_np_arr(np_covariates_X, np_outcomes_Y, axis=1)

        np_treatment_Y = Utils.convert_df_to_np_arr(treatment_Y)

        return np_X, np_treatment_Y

    @staticmethod
    def __preprocess_data_for_DCN(df_X, treatment_index, is_synthetic):
        df = df_X[df_X.iloc[:, -2] == treatment_index]
        # col of X -> x1 .. x17, Y_f, T, Ps
        if is_synthetic:
            # for synthetic dataset #covariates: 75
            df_X = df.iloc[:, 0:75]
        else:
            # for original dataset #covariates: 17
            df_X = df.iloc[:, 0:17]

        ps_score = df.iloc[:, -1]
        df_Y_f = df.iloc[:, -3]
        df_e = df.iloc[:, -4]

        return df_X, ps_score, df_Y_f, df_e

    @staticmethod
    def __convert_to_numpy_DCN(df_X, ps_score, df_Y_f, df_e):
        np_df_X = Utils.convert_df_to_np_arr(df_X)
        np_ps_score = Utils.convert_df_to_np_arr(ps_score)
        np_df_Y_f = Utils.convert_df_to_np_arr(df_Y_f)
        np_df_e = Utils.convert_df_to_np_arr(df_e)
        # print("np_df_X: {0}".format(np_df_X.shape))
        # print("np_ps_score: {0}".format(np_ps_score.shape))
        # print("np_df_Y_f: {0}".format(np_df_Y_f.shape))

        return np_df_X, np_ps_score, np_df_Y_f, np_df_e

    @staticmethod
    def preprocess_dataset_for_training(csv_path):
        data_X = np.loadtxt(csv_path, delimiter=",", skiprows=1)

        # Define features
        x = data_X[:, :30]
        no, dim = x.shape

        # Define potential outcomes
        potential_y = data_X[:, 30:]
        # Die within 1 year = 1, otherwise = 0
        potential_y = np.array(potential_y < 9999, dtype=float)

        # Assign treatment
        coef = np.random.uniform(-0.01, 0.01, size=[dim, 1])
        prob_temp = expit(np.matmul(x, coef) + np.random.normal(0, 0.01, size=[no, 1]))
        prob_t = prob_temp / (2 * np.mean(prob_temp))
        prob_t[prob_t > 1] = 1

        t = np.random.binomial(1, prob_t, [no, 1])
        t = t.reshape([no, ])
        t = Utils.convert_to_col_vector(t)
        y = np.zeros([no, 1])
        y = np.transpose(t) * potential_y[:, 1] + np.transpose(1 - t) * potential_y[:, 0]
        y = np.reshape(np.transpose(y), [no, ])
        y_f = Utils.convert_to_col_vector(y)
        np_X = np.concatenate((x, y_f, potential_y), axis=1)

        print(x.shape)
        print(potential_y.shape)
        print(y_f.shape)
        print(t.shape)

        print(np_X.shape)
        return np_X, t
