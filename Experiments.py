from collections import OrderedDict

import numpy as np

from PM_GAN import PM_GAN
from Utils import Utils
from dataloader import DataLoader


class Experiments:
    def run_all_experiments(self, iterations, running_mode):

        csv_path = "Dataset/ihdp_sample.csv"
        split_size = 0.8
        device = Utils.get_device()
        print(device)
        results_list = []

        run_parameters = self.__get_run_parameters(running_mode)
        file1 = open(run_parameters["summary_file_name"], "a")
        for iter_id in range(iterations):
            print("########### 400 epochs ###########")
            iter_id += 1
            print("--" * 20)
            print("iter_id: {0}".format(iter_id))
            print("--" * 20)
            # load data for propensity network
            dL = DataLoader()

            np_covariates_X_train, np_covariates_X_test, np_covariates_Y_train, np_covariates_Y_test \
                = self.load_data(running_mode, dL, csv_path, split_size)

            pm_gan = PM_GAN()
            pm_gan.train_eval_DCN(iter_id,
                                  np_covariates_X_train,
                                  np_covariates_Y_train,
                                  dL, device, run_parameters,
                                  is_synthetic=run_parameters["is_synthetic"])

            # test DCN network
            reply = pm_gan.test_DCN(iter_id,
                                    np_covariates_X_test,
                                    np_covariates_Y_test,
                                    dL,
                                    device, run_parameters)

            MSE_NN = reply["MSE_NN"]
            true_ATE_NN = reply["true_ATE_NN"]
            predicted_ATE_NN = reply["predicted_ATE_NN"]

            file1.write("Iter: {0}, MSE_NN: {1}\n"
                        .format(iter_id, MSE_NN))
            result_dict = OrderedDict()
            result_dict["iter_id"] = iter_id
            result_dict["MSE_NN"] = MSE_NN

            result_dict["true_ATE_NN"] = true_ATE_NN

            result_dict["predicted_ATE_NN"] = predicted_ATE_NN

            results_list.append(result_dict)

        MSE_set_NN = []

        true_ATE_NN_set = []

        predicted_ATE_NN_set = []

        for result in results_list:
            MSE_set_NN.append(result["MSE_NN"])
            true_ATE_NN_set.append(result["true_ATE_NN"])
            predicted_ATE_NN_set.append(result["predicted_ATE_NN"])

        MSE_total_NN = np.mean(np.array(MSE_set_NN))
        std_MSE_NN = np.std(MSE_set_NN)
        Mean_ATE_NN_true = np.mean(np.array(true_ATE_NN_set))
        std_ATE_NN_true = np.std(true_ATE_NN_set)
        Mean_ATE_NN_predicted = np.mean(np.array(predicted_ATE_NN_set))
        std_ATE_NN_predicted = np.std(predicted_ATE_NN_set)

        print("--" * 20)
        print("Using NN, MSE: {0}, SD: {1}".format(MSE_total_NN, std_MSE_NN))
        print("Using NN, true ATE: {0}, SD: {1}".format(Mean_ATE_NN_true, std_ATE_NN_true))
        print("Using NN, predicted ATE: {0}, SD: {1}".format(Mean_ATE_NN_predicted,
                                                             std_ATE_NN_predicted))
        print("--" * 20)

        file1.write("\n##################################################")
        file1.write("\n")
        file1.write("\nUsing NN, MSE: {0}, SD: {1}".format(MSE_total_NN, std_MSE_NN))
        file1.write("\nUsing NN, true ATE: {0}, SD: {1}".format(Mean_ATE_NN_true,
                                                                std_ATE_NN_true))
        file1.write("\nUsing NN, predicted ATE: {0}, SD: {1}".format(Mean_ATE_NN_predicted,
                                                                     std_ATE_NN_predicted))
        file1.write("\n##################################################")

        Utils.write_to_csv(run_parameters["consolidated_file_path"], results_list)

    @staticmethod
    def __get_run_parameters(running_mode):
        run_parameters = {}
        if running_mode == "original_data":
            run_parameters["input_nodes"] = 25
            run_parameters["consolidated_file_path"] = "./MSE/Results_consolidated.csv"

            # NN
            run_parameters["nn_prop_file"] = "./MSE/NN_Prop_score_{0}.csv"
            run_parameters["nn_iter_file"] = "./MSE/ITE/ITE_NN_iter_{0}.csv"

        elif running_mode == "synthetic_data":
            run_parameters["input_nodes"] = 75
            run_parameters["consolidated_file_path"] = "./MSE_Augmented/Results_consolidated.csv"

            # NN
            run_parameters["nn_prop_file"] = "./MSE_Augmented/NN_Prop_score_{0}.csv"
            run_parameters["nn_iter_file"] = "./MSE_Augmented/ITE/ITE_NN_iter_{0}.csv"
            # SAE
            run_parameters["sae_e2e_prop_file"] = "./MSE_Augmented/SAE_E2E_Prop_score_{0}.csv"
            run_parameters["sae_stacked_all_prop_file"] = "./MSE_Augmented/SAE_stacked_all_Prop_score_{0}.csv"
            run_parameters["sae_stacked_cur_prop_file"] = "./MSE_Augmented/SAE_stacked_cur_Prop_score_{0}.csv"

            run_parameters["sae_e2e_iter_file"] = "./MSE_Augmented/ITE/ITE_SAE_E2E_iter_{0}.csv"
            run_parameters["sae_stacked_all_iter_file"] = "./MSE_Augmented/ITE/ITE_SAE_stacked_all_iter_{0}.csv"
            run_parameters["sae_stacked_cur_iter_file"] = "./MSE_Augmented/ITE/ITE_SAE_stacked_cur_Prop_iter_{0}.csv"

            # LR
            run_parameters["lr_prop_file"] = "./MSE_Augmented/LR_Prop_score_{0}.csv"
            run_parameters["lr_iter_file"] = "./MSE_Augmented/ITE/ITE_LR_iter_{0}.csv"
            # LR Lasso
            run_parameters["lr_prop_file"] = "./MSE_Augmented/LR_lasso_Prop_score_{0}.csv"
            run_parameters["lr_iter_file"] = "./MSE_Augmented/ITE/ITE_LR_Lasso_iter_{0}.csv"
            run_parameters["summary_file_name"] = "Details_augmented.txt"
            run_parameters["is_synthetic"] = True

        return run_parameters

    @staticmethod
    def load_data(running_mode, dL, csv_path, split_size):
        if running_mode == "original_data":
            return dL.preprocess_data_from_csv(csv_path, split_size)

        elif running_mode == "synthetic_data":
            return dL.preprocess_data_from_csv_augmented(csv_path, split_size)
