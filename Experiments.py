from collections import OrderedDict
from datetime import date

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
        print(str(run_parameters["summary_file_name"]))
        # file1 = open(run_parameters["summary_file_name"], "a")
        for iter_id in range(iterations):
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

            MSE_NN_PD = reply["MSE_NN_PD"]
            true_ATE_NN_PD = reply["true_ATE_NN_PD"]
            predicted_ATE_NN_PD = reply["predicted_ATE_NN_PD"]

            MSE_NN_no_dropout = reply["MSE_NN_no_dropout"]
            true_ATE_NN_no_dropout = reply["true_ATE_NN_no_dropout"]
            predicted_ATE_NN_no_dropout = reply["predicted_ATE_NN_no_dropout"]

            MSE_NN_dropout_5 = reply["MSE_NN_dropout_5"]
            true_ATE_NN_dropout_5 = reply["true_ATE_NN_dropout_5"]
            predicted_ATE_NN_dropout_5 = reply["predicted_ATE_NN_dropout_5"]

            MSE_NN_dropout_2 = reply["MSE_NN_dropout_2"]
            true_ATE_NN_dropout_2 = reply["true_ATE_NN_dropout_2"]
            predicted_ATE_NN_dropout_2 = reply["predicted_ATE_NN_dropout_2"]

            # file1.write("\nToday's date: {0}\n".format(date.today()))
            # file1.write("Iter: {0}, MSE_NN_PD (No PS Match): {1}, "
            #             "MSE_NN_no_dropout (PS Match): {2}, "
            #             "MSE_NN_dropout_5 (PS Match): {3}, "
            #             "MSE_NN_dropout_2 (PS Match): {4}\n"
            #             .format(iter_id, MSE_NN_PD, MSE_NN_no_dropout, MSE_NN_dropout_5,
            #                     MSE_NN_dropout_2))
            result_dict = OrderedDict()
            result_dict["iter_id"] = iter_id
            result_dict["MSE_NN_PD"] = MSE_NN_PD
            result_dict["true_ATE_NN_PD"] = true_ATE_NN_PD
            result_dict["predicted_ATE_NN_PD"] = predicted_ATE_NN_PD

            result_dict["MSE_NN_no_dropout"] = MSE_NN_no_dropout
            result_dict["true_ATE_NN_no_dropout"] = true_ATE_NN_no_dropout
            result_dict["predicted_ATE_NN_no_dropout"] = predicted_ATE_NN_no_dropout

            result_dict["MSE_NN_dropout_5"] = MSE_NN_dropout_5
            result_dict["true_ATE_NN_dropout_5"] = true_ATE_NN_dropout_5
            result_dict["predicted_ATE_NN_dropout_5"] = predicted_ATE_NN_dropout_5

            result_dict["MSE_NN_dropout_2"] = MSE_NN_dropout_2
            result_dict["true_ATE_NN_dropout_2"] = true_ATE_NN_dropout_2
            result_dict["predicted_ATE_NN_dropout_2"] = predicted_ATE_NN_dropout_2

            results_list.append(result_dict)

        MSE_set_NN_PD = []
        true_ATE_NN_set_PD = []
        predicted_ATE_NN_set_PD = []

        MSE_set_NN_no_dropout = []
        true_ATE_NN_set_no_dropout = []
        predicted_ATE_NN_set_no_dropout = []

        MSE_set_NN_dropout_5 = []
        true_ATE_NN_set_dropout_5 = []
        predicted_ATE_NN_set_dropout_5 = []

        MSE_set_NN_dropout_2 = []
        true_ATE_NN_set_dropout_2 = []
        predicted_ATE_NN_set_dropout_2 = []

        for result in results_list:
            MSE_set_NN_PD.append(result["MSE_NN_PD"])
            true_ATE_NN_set_PD.append(result["true_ATE_NN_PD"])
            predicted_ATE_NN_set_PD.append(result["predicted_ATE_NN_PD"])

            MSE_set_NN_no_dropout.append(result["MSE_NN_no_dropout"])
            true_ATE_NN_set_no_dropout.append(result["true_ATE_NN_no_dropout"])
            predicted_ATE_NN_set_no_dropout.append(result["predicted_ATE_NN_no_dropout"])

            MSE_set_NN_dropout_5.append(result["MSE_NN_dropout_5"])
            true_ATE_NN_set_dropout_5.append(result["true_ATE_NN_dropout_5"])
            predicted_ATE_NN_set_dropout_5.append(result["predicted_ATE_NN_dropout_5"])

            MSE_set_NN_dropout_2.append(result["MSE_NN_dropout_2"])
            true_ATE_NN_set_dropout_2.append(result["true_ATE_NN_dropout_2"])
            predicted_ATE_NN_set_dropout_2.append(result["predicted_ATE_NN_dropout_2"])

        MSE_mean_NN_PD = np.mean(np.array(MSE_set_NN_PD))
        std_MSE_NN_PD = np.std(MSE_set_NN_PD)
        Mean_ATE_NN_PD_true = np.mean(np.array(true_ATE_NN_set_PD))
        std_ATE_NN_PD_true = np.std(true_ATE_NN_set_PD)
        Mean_ATE_NN_PD_predicted = np.mean(np.array(predicted_ATE_NN_set_PD))
        std_ATE_NN_PD_predicted = np.std(predicted_ATE_NN_set_PD)

        MSE_mean_NN_no_dropout = np.mean(np.array(MSE_set_NN_no_dropout))
        std_MSE_NN_no_dropout = np.std(MSE_set_NN_no_dropout)
        Mean_ATE_NN_no_dropout_true = np.mean(np.array(true_ATE_NN_set_no_dropout))
        std_ATE_NN_no_dropout_true = np.std(true_ATE_NN_set_no_dropout)
        Mean_ATE_NN_no_dropout_predicted = np.mean(np.array(predicted_ATE_NN_set_no_dropout))
        std_ATE_NN_no_dropout_predicted = np.std(predicted_ATE_NN_set_no_dropout)

        MSE_mean_NN_dropout_5 = np.mean(np.array(MSE_set_NN_dropout_5))
        std_MSE_NN_dropout_5 = np.std(MSE_set_NN_dropout_5)
        Mean_ATE_NN_dropout_5_true = np.mean(np.array(true_ATE_NN_set_dropout_5))
        std_ATE_NN_dropout_5_true = np.std(true_ATE_NN_set_dropout_5)
        Mean_ATE_NN_dropout_5_predicted = np.mean(np.array(predicted_ATE_NN_set_dropout_5))
        std_ATE_NN_dropout_5_predicted = np.std(predicted_ATE_NN_set_dropout_5)

        MSE_mean_NN_dropout_2 = np.mean(np.array(MSE_set_NN_dropout_2))
        std_MSE_NN_dropout_2 = np.std(MSE_set_NN_dropout_2)
        Mean_ATE_NN_dropout_2_true = np.mean(np.array(true_ATE_NN_set_dropout_2))
        std_ATE_NN_dropout_2_true = np.std(true_ATE_NN_set_dropout_2)
        Mean_ATE_NN_dropout_2_predicted = np.mean(np.array(predicted_ATE_NN_set_dropout_2))
        std_ATE_NN_dropout_2_predicted = np.std(predicted_ATE_NN_set_dropout_2)

        print("--" * 20)
        print("Using NN PD (No PS Match), MSE: {0}, SD: {1}"
              .format(MSE_mean_NN_PD, std_MSE_NN_PD))
        print("Using NN PD (No PS Match), true ATE: {0}, SD: {1}"
              .format(Mean_ATE_NN_PD_true, std_ATE_NN_PD_true))
        print("Using NN PD (No PS Match), predicted ATE: {0}, SD: {1}"
              .format(Mean_ATE_NN_PD_predicted,
                      std_ATE_NN_PD_predicted))
        print("--" * 20)

        print("Using NN No Dropout (PS Match), MSE: {0}, SD: {1}"
              .format(MSE_mean_NN_no_dropout, std_MSE_NN_no_dropout))
        print("Using NN No Dropout (PS Match), true ATE: {0}, SD: {1}"
              .format(Mean_ATE_NN_no_dropout_true, std_ATE_NN_no_dropout_true))
        print("Using NN No Dropout (PS Match), predicted ATE: {0}, SD: {1}"
              .format(Mean_ATE_NN_no_dropout_predicted,
                      std_ATE_NN_no_dropout_predicted))
        print("--" * 20)

        print("Using NN Dropout 0.5 (PS Match), MSE: {0}, SD: {1}"
              .format(MSE_mean_NN_dropout_5, std_MSE_NN_dropout_5))
        print("Using NN Dropout 0.5 (PS Match), true ATE: {0}, SD: {1}"
              .format(Mean_ATE_NN_dropout_5_true, std_ATE_NN_dropout_5_true))
        print("Using NN Dropout 0.5 (PS Match), predicted ATE: {0}, SD: {1}"
              .format(Mean_ATE_NN_dropout_5_predicted,
                      std_ATE_NN_dropout_5_predicted))
        print("--" * 20)

        print("Using NN PD Dropout 0.2 (PS Match), MSE: {0}, SD: {1}"
              .format(MSE_mean_NN_dropout_2, std_MSE_NN_dropout_2))
        print("Using NN Dropout 0.2 (PS Match), true ATE: {0}, SD: {1}"
              .format(Mean_ATE_NN_dropout_2_true, std_ATE_NN_dropout_2_true))
        print("Using NN Dropout 0.2 (PS Match), predicted ATE: {0}, SD: {1}"
              .format(Mean_ATE_NN_dropout_2_predicted,
                      std_ATE_NN_dropout_2_predicted))
        print("--" * 20)

        # file1.write("\n##################################################")
        # file1.write("\n")
        # file1.write("\nUsing NN PD (No PS Match), MSE: {0}, SD: {1}"
        #             .format(MSE_mean_NN_PD, std_MSE_NN_PD))
        # file1.write("\nUsing NN PD (No PS Match), true ATE: {0}, SD: {1}"
        #             .format(Mean_ATE_NN_PD_true,
        #                     std_ATE_NN_PD_true))
        # file1.write("\nUsing NN PD (No PS Match), predicted ATE: {0}, SD: {1}"
        #             .format(Mean_ATE_NN_PD_predicted,
        #                     std_ATE_NN_PD_predicted))
        # file1.write("\n##################################################")
        #
        # file1.write("\n##################################################")
        # file1.write("\n")
        # file1.write("\nUsing NN No Dropout (PS Match), MSE: {0}, SD: {1}"
        #             .format(MSE_mean_NN_no_dropout, std_MSE_NN_no_dropout))
        # file1.write("\nUsing NN No Dropout (PS Match), true ATE: {0}, SD: {1}"
        #             .format(Mean_ATE_NN_no_dropout_true,
        #                     std_ATE_NN_no_dropout_true))
        # file1.write("\nUsing NN No Dropout (PS Match), predicted ATE: {0}, SD: {1}"
        #             .format(Mean_ATE_NN_no_dropout_predicted,
        #                     std_ATE_NN_no_dropout_predicted))
        # file1.write("\n##################################################")
        #
        # file1.write("\n##################################################")
        # file1.write("\n")
        # file1.write("\nUsing NN Dropout 0.5 (PS Match), MSE: {0}, SD: {1}"
        #             .format(MSE_mean_NN_dropout_5, std_MSE_NN_dropout_5))
        # file1.write("\nUsing NN Dropout 0.5 (PS Match), true ATE: {0}, SD: {1}"
        #             .format(Mean_ATE_NN_dropout_5_true,
        #                     std_ATE_NN_dropout_5_true))
        # file1.write("\nUsing NN Dropout 0.5 (PS Match), predicted ATE: {0}, SD: {1}"
        #             .format(Mean_ATE_NN_dropout_5_predicted,
        #                     std_ATE_NN_dropout_5_predicted))
        # file1.write("\n##################################################")
        #
        # file1.write("\n##################################################")
        # file1.write("\n")
        # file1.write("\nUsing NN Dropout 0.2 (PS Match), MSE: {0}, SD: {1}"
        #             .format(MSE_mean_NN_dropout_2, std_MSE_NN_dropout_2))
        # file1.write("\nUsing NN Dropout 0.2 (PS Match), true ATE: {0}, SD: {1}"
        #             .format(Mean_ATE_NN_dropout_2_true,
        #                     std_ATE_NN_dropout_2_true))
        # file1.write("\nUsing NN Dropout 0.2 (PS Match), predicted ATE: {0}, SD: {1}"
        #             .format(Mean_ATE_NN_dropout_2_predicted,
        #                     std_ATE_NN_dropout_2_predicted))
        # file1.write("\n##################################################")

        Utils.write_to_csv(run_parameters["consolidated_file_path"], results_list)

    @staticmethod
    def __get_run_parameters(running_mode):
        run_parameters = {}
        if running_mode == "original_data":
            run_parameters["input_nodes"] = 25
            run_parameters["consolidated_file_path"] = "./MSE/Results_consolidated.csv"

            # NN
            run_parameters["nn_prop_file"] = "./MSE/NN_Prop_score_{0}.csv"

            run_parameters["nn_DCN_PD_iter_file"] = "./MSE/ITE/ITE_NN_DCN_PD_No_PS_Match_iter_{0}.csv"
            run_parameters["nn_DCN_No_Dropout_iter_file"] = "./MSE/ITE/ITE_NN_DCN_No_dropout_iter_{0}.csv"
            run_parameters["nn_DCN_Const_Dropout_5_iter_file"] = "./MSE/ITE/ITE_NN_DCN_Const_dropout_5_iter_{0}.csv"
            run_parameters["nn_DCN_Const_Dropout_2_iter_file"] = "./MSE/ITE/ITE_NN_DCN_Const_dropout_2_iter_{0}.csv"

            run_parameters["summary_file_name"] = "Details_original.txt"
            run_parameters["is_synthetic"] = False

        elif running_mode == "synthetic_data":
            run_parameters["input_nodes"] = 75
            run_parameters["consolidated_file_path"] = "./MSE_Augmented/Results_consolidated.csv"

            # NN
            run_parameters["nn_prop_file"] = "./MSE_Augmented/NN_Prop_score_{0}.csv"

            run_parameters["nn_DCN_PD_iter_file"] = "./MSE_Augmented/ITE/ITE_NN_DCN_PD_No_PS_Match_iter_{0}.csv"
            run_parameters["nn_DCN_No_Dropout_iter_file"] = "./MSE_Augmented/ITE/ITE_NN_DCN_No_dropout_iter_{0}.csv"
            run_parameters[
                "nn_DCN_Const_Dropout_5_iter_file"] = "./MSE_Augmented/ITE/ITE_NN_DCN_Const_dropout_5_iter_{0}.csv"
            run_parameters[
                "nn_DCN_Const_Dropout_2_iter_file"] = "./MSE_Augmented/ITE/ITE_NN_DCN_Const_dropout_2_iter_{0}.csv"

            run_parameters["summary_file_name"] = "Details_augmented.txt"
            run_parameters["is_synthetic"] = True

        return run_parameters

    @staticmethod
    def load_data(running_mode, dL, csv_path, split_size):
        if running_mode == "original_data":
            return dL.preprocess_data_from_csv(csv_path, split_size)

        elif running_mode == "synthetic_data":
            return dL.preprocess_data_from_csv_augmented(csv_path, split_size)
