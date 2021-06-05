import os

import numpy as np
import pandas as pd

from Utils import Utils


class DataLoader:
    def preprocess_for_graphs(self, csv_path):
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), csv_path), header=None)
        return self.__convert_to_numpy_1(df)

    def prep_process_all_data(self, csv_path):
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), csv_path), header=None)
        np_covariates_X, np_treatment_Y = self.__convert_to_numpy_1(df)
        return np_covariates_X, np_treatment_Y

    def preprocess_data_from_csv(self, csv_path, split_size):
        # print(".. Data Loading ..")
        # data load
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), csv_path), header=None)
        np_covariates_X, np_treatment_T, np_outcomes_Y_f, np_outcomes_Y_cf, \
        np_mu0, np_mu1 = self.__convert_to_numpy(df)
        print("ps_np_covariates_X: {0}".format(np_covariates_X.shape))
        print("ps_np_treatment_Y: {0}".format(np_treatment_T.shape))

        np_train_X, np_test_X, np_train_T, np_test_T, np_train_yf, np_test_yf, np_train_ycf, np_test_ycf, \
        np_train_mu0, np_test_mu0, np_train_mu1, np_test_mu1 = \
            Utils.test_train_split(np_covariates_X, np_treatment_T, np_outcomes_Y_f,
                                   np_outcomes_Y_cf, np_mu0, np_mu1, split_size)

        np_train_X, np_val_X, np_train_T, np_val_T, np_train_yf, np_val_yf, \
        np_train_ycf, np_val_ycf, \
        np_train_mu0, np_val_mu0, np_train_mu1, np_val_mu1 = \
            Utils.test_train_split(np_train_X, np_train_T, np_train_yf,
                                   np_train_ycf, np_train_mu0, np_train_mu1, split_size=0.90)

        print("Numpy Train Statistics:")
        print(np_train_X.shape)
        print(np_train_T.shape)
        n_treated = np_train_T[np_train_T == 1]
        # print(n_treated.shape[0])
        # print(np_train_T.shape[0])

        n_treated = n_treated.shape[0]
        n_total = np_train_T.shape[0]
        print("Numpy Val Statistics:")
        print(np_val_X.shape)
        print(np_val_T.shape)
        print(np_val_yf.shape)
        print(np_val_ycf.shape)

        print("Numpy Temp Statistics:")
        print(np_test_X.shape)
        print(np_test_T.shape)

        return np_train_X, np_train_T, np_train_yf, np_train_ycf, \
               np_test_X, np_test_T, np_test_yf, np_test_ycf, \
               np_val_X, np_val_T, np_val_yf, np_val_ycf, \
               np_train_mu0, np_test_mu0, np_train_mu1, np_test_mu1, \
               np_val_mu0, np_val_mu1

    def prepare_tensor_for_ITE(self, np_X,
                               np_T,
                               np_yf,
                               np_ycf,
                               np_mu0,
                               np_mu1,
                               ps_list):
        print(np_X.shape)
        print(np_T.shape)
        print("----------")
        # col of X -> x1 .. x25, Y_f, Y_cf, mu_0, mu_1, T, Ps
        X = np.concatenate((np_X, np_yf, np_ycf, np_mu0,
                            np_mu1, np_T, np.array([ps_list]).T), axis=1)
        print(X.shape)

        df_X = pd.DataFrame(X)
        treated_df_X, treated_ps_score, treated_df_Y_f, treated_df_Y_cf, \
        treated_df_mu_0, treated_df_mu_1 = self.__preprocess_data_for_DCN(df_X,
                                                                          treatment_index=1)

        control_df_X, control_ps_score, control_df_Y_f, control_df_Y_cf, \
        control_df_mu_0, control_df_mu_1 = self.__preprocess_data_for_DCN(df_X,
                                                                          treatment_index=0)

        np_treated_df_X, np_treated_ps_score, np_treated_df_Y_f, np_treated_df_Y_cf, \
        np_treated_df_mu_0, np_treated_df_mu_1 = \
            self.__convert_to_numpy_DCN(treated_df_X, treated_ps_score, treated_df_Y_f,
                                        treated_df_Y_cf, treated_df_mu_0, treated_df_mu_1)

        np_control_df_X, np_control_ps_score, np_control_df_Y_f, np_control_df_Y_cf, \
        np_control_df_mu_0, np_control_df_mu_1 = \
            self.__convert_to_numpy_DCN(control_df_X, control_ps_score, control_df_Y_f,
                                        control_df_Y_cf, control_df_mu_0, control_df_mu_1)

        print(" Treated Statistics ==>")
        print(np_treated_df_X.shape)
        # print(np_treated_ps_score.shape)

        print(" Control Statistics ==>")
        print(np_control_df_X.shape)
        # print(np_control_ps_score.shape)

        return {
            "treated_data": (np_treated_df_X, np_treated_ps_score,
                             np_treated_df_Y_f, np_treated_df_Y_cf,
                             np_treated_df_mu_0, np_treated_df_mu_1),
            "control_data": (np_control_df_X, np_control_ps_score,
                             np_control_df_Y_f, np_control_df_Y_cf,
                             np_control_df_mu_0, np_control_df_mu_1)
        }

    @staticmethod
    def __convert_to_numpy_DCN(df_X, ps_score, df_Y_f, df_Y_cf, mu_0, mu_1):
        np_df_X = Utils.convert_df_to_np_arr(df_X)
        np_ps_score = Utils.convert_df_to_np_arr(ps_score)
        np_df_Y_f = Utils.convert_df_to_np_arr(df_Y_f)
        np_df_Y_cf = Utils.convert_df_to_np_arr(df_Y_cf)
        np_mu_0 = Utils.convert_df_to_np_arr(mu_0)
        np_mu_1 = Utils.convert_df_to_np_arr(mu_1)

        return np_df_X, np_ps_score, np_df_Y_f, np_df_Y_cf, np_mu_0, np_mu_1

    @staticmethod
    def __preprocess_data_for_DCN(df_X, treatment_index):
        # col of X -> x1 .. x25, Y_f, Y_cf, mu_0, mu_1, T, Ps
        df = df_X[df_X.iloc[:, -2] == treatment_index]

        df_X = df.iloc[:, 0:25]

        ps_score = df.iloc[:, -1]
        df_Y_f = df.iloc[:, -6]
        df_Y_cf = df.iloc[:, -5]
        df_mu_0 = df.iloc[:, -4]
        df_mu_1 = df.iloc[:, -3]

        return df_X, ps_score, df_Y_f, df_Y_cf, df_mu_0, df_mu_1

    @staticmethod
    def __convert_to_numpy(df):
        covariates_X = df.iloc[:, 5:]
        treatment_T = df.iloc[:, 0]
        outcomes_Y_f = df.iloc[:, 1]
        outcomes_Y_cf = df.iloc[:, 2]
        mu0 = df.iloc[:, 3]
        mu1 = df.iloc[:, 4]

        np_covariates_X = Utils.convert_df_to_np_arr(covariates_X)
        np_outcomes_Y_f = Utils.convert_to_col_vector(Utils.convert_df_to_np_arr(outcomes_Y_f))
        np_outcomes_Y_cf = Utils.convert_to_col_vector(Utils.convert_df_to_np_arr(outcomes_Y_cf))
        np_treatment_T = Utils.convert_to_col_vector(Utils.convert_df_to_np_arr(treatment_T))
        np_mu0 = Utils.convert_to_col_vector(Utils.convert_df_to_np_arr(mu0))
        np_mu1 = Utils.convert_to_col_vector(Utils.convert_df_to_np_arr(mu1))

        return np_covariates_X, np_treatment_T, np_outcomes_Y_f, np_outcomes_Y_cf, np_mu0, np_mu1

    @staticmethod
    def __convert_to_numpy_1(df):
        covariates_X = df.iloc[:, 5:]
        treatment_Y = df.iloc[:, 0:1]
        outcomes_Y = df.iloc[:, 1:3]

        np_covariates_X = Utils.convert_df_to_np_arr(covariates_X)
        np_outcomes_Y = Utils.convert_df_to_np_arr(outcomes_Y)
        np_X = Utils.concat_np_arr(np_covariates_X, np_outcomes_Y, axis=1)

        np_treatment_Y = Utils.convert_df_to_np_arr(treatment_Y)

        return np_X, np_treatment_Y
