from collections import OrderedDict
from datetime import date

import numpy as np

from Constants import Constants
from DCN_Experiments import DCN_Experiments
from Metrics import Metrics
from PS_Manager import PS_Manager
from PS_Treated_Generator import PS_Treated_Generator
from TARNet_Experiments import TARNet_Experiments
from Utils import Utils
from dataloader import DataLoader


class Experiments:
    def __init__(self, running_mode, csv_path, split_size):
        self.dL = DataLoader()
        self.running_mode = running_mode
        self.np_covariates_X_train, self.np_treatment_Y \
            = self.__load_data(csv_path, split_size)

    def run_all_experiments(self, iterations, ps_model_type):
        device = Utils.get_device()
        print(device)
        results_list = []

        run_parameters = self.__get_run_parameters()
        print(str(run_parameters["summary_file_name"]))
        file1 = open(run_parameters["summary_file_name"], "a")
        for iter_id in range(iterations):
            iter_id += 1
            print("--" * 20)
            print("iter_id: {0}".format(iter_id))
            print("Bacteria")
            print("--" * 20)
            input_nodes = run_parameters["input_nodes"]

            # get propensity score for classifier training and testing
            ps_score_list_train, ps_model = \
                self.__get_ps_model(ps_model_type,
                                    iter_id,
                                    run_parameters[
                                        "input_nodes"],
                                    device)
            run_parameters["consolidated_file_path"] = self.get_consolidated_file_name(ps_model_type)

            data_loader_dict_train = self.dL.prepare_tensor_for_DCN(self.np_covariates_X_train,
                                                                    self.np_treatment_Y,
                                                                    ps_score_list_train,
                                                                    run_parameters["is_synthetic"])
            # tensor_treated_train_original = \
            #     Utils.create_tensors_from_tuple(data_loader_dict_train["treated_data"])
            # tensor_control_train_original = \
            #     Utils.create_tensors_from_tuple(data_loader_dict_train["control_data"])
            #
            # n_treated_original = data_loader_dict_train["treated_data"][0].shape[0]
            # n_control_original = data_loader_dict_train["control_data"][0].shape[0]
            #
            # Execute PM GAN
            ps_t = PS_Treated_Generator(data_loader_dict_train,
                                        ps_model, ps_model_type)

            ps_t.simulate_treated_semi_supervised(input_nodes, iter_id, device)


    def __get_run_parameters(self):
        run_parameters = {}
        if self.running_mode == "original_data":
            run_parameters["input_nodes"] = 3198
            # run_parameters["consolidated_file_path"] = "./MSE/Results_consolidated.csv"

            # NN
            run_parameters["nn_prop_file"] = "./MSE/NN_Prop_score_{0}.csv"

            run_parameters["DCN_PD"] = "./MSE/ITE/ITE_DCN_PD_iter_{0}.csv"
            run_parameters["DCN_PD_02"] = "./MSE/ITE/ITE_DCN_PD_02_iter_{0}.csv"
            run_parameters["DCN_PD_05"] = "./MSE/ITE/ITE_DCN_PD_05_iter_{0}.csv"

            run_parameters["DCN_PM_GAN"] = "./MSE/ITE/ITE_DCN_PM_GAN_iter_{0}.csv"
            run_parameters["DCN_PM_GAN_02"] = "./MSE/ITE/ITE_DCN_PM_GAN_dropout_02_iter_{0}.csv"
            run_parameters["DCN_PM_GAN_05"] = "./MSE/ITE/ITE_DCN_PM_GAN_dropout_05_iter_{0}.csv"
            run_parameters["DCN_PM_GAN_PD"] = "./MSE/ITE/ITE_DCN_PM_GAN_dropout_PD_iter_{0}.csv"

            run_parameters["TARNET"] = "./MSE/ITE/ITE_TARNET_iter_{0}.csv"

            run_parameters["TARNET_PM_GAN"] = "./MSE/ITE/ITE_TARNET_PM_GAN_iter_{0}.csv"

            run_parameters["summary_file_name"] = "Details_original.txt"
            run_parameters["is_synthetic"] = False

        elif self.running_mode == "synthetic_data":
            run_parameters["input_nodes"] = 75
            # run_parameters["consolidated_file_path"] = "./MSE_Augmented/Results_consolidated.csv"

            run_parameters["is_synthetic"] = True

        return run_parameters

    def __load_data(self, csv_path, split_size):
        if self.running_mode == "original_data":
            return self.dL.preprocess_data_from_csv(csv_path, split_size)

        # elif self.running_mode == "synthetic_data":
        #     return self.dL.preprocess_data_from_csv_augmented(csv_path, split_size)

    def __get_ps_model(self, ps_model_type, iter_id,
                       input_nodes, device):
        ps_train_set = self.dL.convert_to_tensor(self.np_covariates_X_train,
                                                 self.np_treatment_Y)
        ps_manager = PS_Manager()
        if ps_model_type == Constants.PS_MODEL_NN:
            return ps_manager.get_propensity_scores(ps_train_set,
                                                    iter_id,
                                                    input_nodes, device)

        # elif ps_model_type == Constants.PS_MODEL_LR:
        #     return ps_manager.get_propensity_scores_using_LR(self.np_covariates_X_train,
        #                                                      self.np_covariates_Y_train,
        #                                                      self.np_covariates_X_val,
        #                                                      self.np_covariates_X_test,
        #                                                      regularized=False)
        # elif ps_model_type == Constants.PS_MODEL_LR_Lasso:
        #     return ps_manager.get_propensity_scores_using_LR(self.np_covariates_X_train,
        #                                                      self.np_covariates_Y_train,
        #                                                      self.np_covariates_X_val,
        #                                                      self.np_covariates_X_test,
        #                                                      regularized=True)

    @staticmethod
    def __process_evaluated_metric(y1_true, y0_true, y1_hat, y0_hat,
                                   ite_dict, true_ITE_list, predicted_ITE_list, ite_csv_path, iter_id):
        y1_true_np = np.array(y1_true)
        y0_true_np = np.array(y0_true)
        y1_hat_np = np.array(y1_hat)
        y0_hat_np = np.array(y0_hat)

        PEHE = Metrics.PEHE(y1_true_np, y0_true_np, y1_hat_np, y0_hat_np)
        ATE = Metrics.ATE(y1_true_np, y0_true_np, y1_hat_np, y0_hat_np)
        print("PEHE: {0}".format(PEHE))
        print("ATE: {0}".format(ATE))

        true_ATE = sum(true_ITE_list) / len(true_ITE_list)
        predicted_ATE = sum(predicted_ITE_list) / len(predicted_ITE_list)

        Utils.write_to_csv(ite_csv_path.format(iter_id), ite_dict)
        return PEHE, ATE, true_ATE, predicted_ATE

    def get_consolidated_file_name(self, ps_model_type):
        if ps_model_type == Constants.PS_MODEL_NN:
            return "./MSE/Results_consolidated_NN.csv"
        elif ps_model_type == Constants.PS_MODEL_LR:
            return "./MSE/Results_consolidated_LR.csv"
        elif ps_model_type == Constants.PS_MODEL_LR_Lasso:
            return "./MSE/Results_consolidated_LR_LAsso.csv"
