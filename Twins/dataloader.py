import os

import numpy as np
import pandas as pd
from scipy.special import expit

from Utils import Utils


class DataLoader:
    def preprocess_for_graphs(self, csv_path):
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), csv_path), header=None)
        return self.__convert_to_numpy(df)

    def preprocess_data_from_csv(self, csv_path, split_size):
        np_covariates_X, np_treatment_Y = self.preprocess_dataset_for_training(csv_path)

        np_covariates_X_train, np_covariates_X_test, np_covariates_Y_train, np_covariates_Y_test = \
            Utils.test_train_split(np_covariates_X, np_treatment_Y, split_size)

        # print("np_covariates_X_train: {0}".format(np_covariates_X_train.shape))
        # print("np_covariates_Y_train: {0}".format(np_covariates_Y_train.shape))
        # print("---" * 20)
        # print("np_covariates_X_test: {0}".format(np_covariates_X_test.shape))
        # print("np_covariates_Y_test: {0}".format(np_covariates_Y_test.shape))
        # print("---" * 20)

        return np_covariates_X_train, np_covariates_X_test, np_covariates_Y_train, np_covariates_Y_test

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
        # ps_np_covariates_X -> X1 .. X30, y_f, Y_0, Y_1
        # print("ps_np_covariates_X: {0}".format(ps_np_covariates_X.shape))
        # print("ps_np_treatment_Y: {0}".format(ps_np_treatment_Y.shape))
        X = Utils.concat_np_arr(ps_np_covariates_X, ps_np_treatment_T, axis=1)

        # col of X -> x1 .. x30, Y_f, Y(0), Y(1), T, Ps
        X = Utils.concat_np_arr(X, np.array([ps_list]).T, axis=1)
        print("Big X: {0}".format(X.shape))

        df_X = pd.DataFrame(X)
        treated_df_X, treated_ps_score, treated_df_Y_f, treated_df_Y_0, treated_df_Y_1 = \
            self.__preprocess_data_for_DCN(df_X, treatment_index=1,
                                           is_synthetic=is_synthetic)

        control_df_X, control_ps_score, control_df_Y_f, control_df_Y_0, control_df_Y_1 = \
            self.__preprocess_data_for_DCN(df_X, treatment_index=0,
                                           is_synthetic=is_synthetic)

        np_treated_df_X, np_treated_ps_score, np_treated_df_Y_f, np_treated_df_Y_0, \
        np_treated_df_Y_1 = \
            self.__convert_to_numpy_DCN(treated_df_X, treated_ps_score, treated_df_Y_f,
                                        treated_df_Y_0, treated_df_Y_1)

        np_control_df_X, np_control_ps_score, np_control_df_Y_f, np_control_df_Y_0, \
        np_control_df_Y_1 = \
            self.__convert_to_numpy_DCN(control_df_X, control_ps_score, control_df_Y_f,
                                        control_df_Y_0, control_df_Y_1)

        # np_treated_df_Y_f = Utils.convert_to_col_vector()

        print(" Treated Statistics ==>")
        print(np_treated_df_X.shape)
        print(" Control Statistics ==>")
        print(np_control_df_X.shape)
        # print(np_control_ps_score.shape)

        return {
            "treated_data": (np_treated_df_X, np_treated_ps_score,
                             np_treated_df_Y_f, np_treated_df_Y_0, np_treated_df_Y_1),
            "control_data": (np_control_df_X, np_control_ps_score,
                             np_control_df_Y_f, np_control_df_Y_0, np_control_df_Y_0)
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
        # X -> x1 .. x30, Y_f, Y(0), Y(1), T, Ps
        if is_synthetic:
            # for synthetic dataset #covariates: 75
            df_X = df.iloc[:, 0:75]
        else:
            # for original dataset #covariates: 25
            df_X = df.iloc[:, 0:30]

        ps_score = df.iloc[:, -1]
        df_Y_f = df.iloc[:, -5]
        df_Y_0 = df.iloc[:, -4]
        df_Y_1 = df.iloc[:, -3]

        return df_X, ps_score, df_Y_f, df_Y_0, df_Y_1

    @staticmethod
    def __convert_to_numpy_DCN(df_X, ps_score, df_Y_f, df_Y_0, df_Y_1):
        np_df_X = Utils.convert_df_to_np_arr(df_X)
        np_ps_score = Utils.convert_df_to_np_arr(ps_score)
        np_df_Y_f = Utils.convert_df_to_np_arr(df_Y_f)

        np_df_Y_0 = Utils.convert_df_to_np_arr(df_Y_0)
        np_df_Y_1 = Utils.convert_df_to_np_arr(df_Y_1)

        # print("np_df_X: {0}".format(np_df_X.shape))
        # print("np_ps_score: {0}".format(np_ps_score.shape))
        # print("np_df_Y_f: {0}".format(np_df_Y_f.shape))
        # print("np_df_Y_cf: {0}".format(np_df_Y_cf.shape))

        return np_df_X, np_ps_score, np_df_Y_f, np_df_Y_0, np_df_Y_1
        #
        # return np_df_X, Utils.convert_to_col_vector(np_ps_score), \
        #        Utils.convert_to_col_vector(np_df_Y_f), \
        #        Utils.convert_to_col_vector(np_df_Y_0), \
        #        Utils.convert_to_col_vector(np_df_Y_1)

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
