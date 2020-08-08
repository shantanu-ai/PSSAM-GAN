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
        file1 = open(run_parameters["summary_file_name"], "a")
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

            # train DCN network
            pm_gan.train_DCN(iter_id,
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

            MSE_PM_Match_No_PD = reply["MSE_PM_Match_No_PD"]
            true_ATE_PM_Match_No_PD = reply["true_ATE_PM_Match_No_PD"]
            predicted_ATE_PM_Match_No_PD = reply["predicted_ATE_PM_Match_No_PD"]

            MSE_PM_Match_PD = reply["MSE_PM_Match_PD"]
            true_ATE_PM_Match_PD = reply["true_ATE_PM_Match_PD"]
            predicted_ATE_NN_no_dropout = reply["predicted_ATE_PM_Match_PD"]

            MSE_No_PM_Match_PD = reply["MSE_No_PM_Match_PD"]
            true_ATE_No_PM_Match_PD = reply["true_ATE_No_PM_Match_PD"]
            predicted_ATE_No_PM_Match_PD = reply["predicted_ATE_No_PM_Match_PD"]

            file1.write("\nToday's date: {0}\n".format(date.today()))
            file1.write("Iter: {0}, MSE_NN_PM_PD(GAN): {1}, MSE_NN_DCN_PD: {2}\n"
                        .format(iter_id, MSE_PM_Match_PD, MSE_No_PM_Match_PD))

            result_dict = OrderedDict()
            result_dict["iter_id"] = iter_id
            result_dict["MSE_PM_Match_No_PD"] = MSE_PM_Match_No_PD
            result_dict["true_ATE_PM_Match_No_PD"] = true_ATE_PM_Match_No_PD
            result_dict["predicted_ATE_PM_Match_No_PD"] = predicted_ATE_PM_Match_No_PD

            result_dict["MSE_PM_Match_PD"] = MSE_PM_Match_PD
            result_dict["true_ATE_PM_Match_PD"] = true_ATE_PM_Match_PD
            result_dict["predicted_ATE_NN_no_dropout"] = predicted_ATE_NN_no_dropout

            result_dict["MSE_No_PM_Match_PD"] = MSE_No_PM_Match_PD
            result_dict["true_ATE_No_PM_Match_PD"] = true_ATE_No_PM_Match_PD
            result_dict["predicted_ATE_No_PM_Match_PD"] = predicted_ATE_No_PM_Match_PD
            results_list.append(result_dict)

        MSE_set_PM_Match_No_PD = []
        true_ATE_set_PM_Match_No_PD = []
        predicted_ATE_set_PM_Match_No_PD = []

        MSE_set_PM_Match_PD = []
        true_ATE_set_PM_Match_PD = []
        predicted_ATE_set_PM_Match_PD = []

        MSE_set_No_PM_Match_PD = []
        true_ATE_set_No_PM_Match_PD = []
        predicted_ATE_No_PM_Match_PD = []

        for result in results_list:
            MSE_set_PM_Match_No_PD.append(result["MSE_PM_Match_No_PD"])
            true_ATE_set_PM_Match_No_PD.append(result["true_ATE_PM_Match_No_PD"])
            predicted_ATE_set_PM_Match_No_PD.append(result["predicted_ATE_PM_Match_No_PD"])

            MSE_set_PM_Match_PD.append(result["MSE_PM_Match_PD"])
            true_ATE_set_PM_Match_PD.append(result["true_ATE_PM_Match_PD"])
            predicted_ATE_set_PM_Match_PD.append(result["predicted_ATE_NN_no_dropout"])

            MSE_set_No_PM_Match_PD.append(result["MSE_No_PM_Match_PD"])
            true_ATE_set_No_PM_Match_PD.append(result["true_ATE_No_PM_Match_PD"])
            predicted_ATE_No_PM_Match_PD.append(result["predicted_ATE_No_PM_Match_PD"])

        MSE_mean_PM_Match_No_PD = np.mean(np.array(MSE_set_PM_Match_No_PD))
        std_MSE_PM_Match_No_PD = np.std(MSE_set_PM_Match_No_PD)
        Mean_ATE_PM_Match_No_PD_true = np.mean(np.array(true_ATE_set_PM_Match_No_PD))
        std_ATE_PM_Match_No_PD_true = np.std(true_ATE_set_PM_Match_No_PD)
        Mean_ATE_PM_Match_No_PD_predicted = np.mean(np.array(predicted_ATE_set_PM_Match_No_PD))
        std_ATE_PM_Match_No_PD_predicted = np.std(predicted_ATE_set_PM_Match_No_PD)

        MSE_mean_PM_Match_PD = np.mean(np.array(MSE_set_PM_Match_PD))
        std_MSE_PM_Match_PD = np.std(MSE_set_PM_Match_PD)
        Mean_ATE_PM_Match_PD_true = np.mean(np.array(true_ATE_set_PM_Match_PD))
        std_ATE_PM_Match_PD_true = np.std(true_ATE_set_PM_Match_PD)
        Mean_ATE_PM_Match_PD_predicted = np.mean(np.array(predicted_ATE_set_PM_Match_PD))
        std_ATE_PM_Match_PD_predicted = np.std(predicted_ATE_set_PM_Match_PD)

        MSE_mean_No_PM_Match_PD = np.mean(np.array(MSE_set_No_PM_Match_PD))
        std_MSE_No_PM_Match_PD = np.std(MSE_set_No_PM_Match_PD)
        Mean_ATE_No_PM_Match_PD_true = np.mean(np.array(true_ATE_set_No_PM_Match_PD))
        std_ATE_No_PM_Match_PD_true = np.std(true_ATE_set_No_PM_Match_PD)
        Mean_ATE_No_PM_Match_PD_predicted = np.mean(np.array(predicted_ATE_No_PM_Match_PD))
        std_ATE_No_PM_Match_PD_predicted = np.std(predicted_ATE_No_PM_Match_PD)

        print("--" * 20)
        print("Using PM Match no PD, MSE: {0}, SD: {1}"
              .format(MSE_mean_PM_Match_No_PD, std_MSE_PM_Match_No_PD))
        print("Using PM Match no PD, true ATE: {0}, SD: {1}"
              .format(Mean_ATE_PM_Match_No_PD_true, std_ATE_PM_Match_No_PD_true))
        print("Using PM Match no PD, predicted ATE: {0}, SD: {1}"
              .format(Mean_ATE_PM_Match_No_PD_predicted,
                      std_ATE_PM_Match_No_PD_predicted))
        print("--" * 20)

        print("Using PM Match PD(GAN), MSE: {0}, SD: {1}"
              .format(MSE_mean_PM_Match_PD, std_MSE_PM_Match_PD))
        print("Using PM Match PD(GAN), true ATE: {0}, SD: {1}"
              .format(Mean_ATE_PM_Match_PD_true, std_ATE_PM_Match_PD_true))
        print("Using PM Match PD(GAN), predicted ATE: {0}, SD: {1}"
              .format(Mean_ATE_PM_Match_PD_predicted,
                      std_ATE_PM_Match_PD_predicted))
        print("--" * 20)
        print("DCN - PD")
        print("Using No PM Match PD(DCN_PD), MSE: {0}, SD: {1}"
              .format(MSE_mean_No_PM_Match_PD, std_MSE_No_PM_Match_PD))
        print("Using No PM Match PD(DCN_PD), true ATE: {0}, SD: {1}"
              .format(Mean_ATE_No_PM_Match_PD_true, std_ATE_No_PM_Match_PD_true))
        print("Using No PM Match PD(DCN_PD), predicted ATE: {0}, SD: {1}"
              .format(Mean_ATE_No_PM_Match_PD_predicted,
                      std_ATE_No_PM_Match_PD_predicted))
        print("--" * 20)

        file1.write("\n##################################################")
        file1.write("\n")
        file1.write("\nUsing NN PM PD(GAN), MSE: {0}, SD: {1}"
                    .format(MSE_mean_PM_Match_PD, std_MSE_PM_Match_PD))
        file1.write("\nUsing NN PM PD(GAN), true ATE: {0}, SD: {1}"
                    .format(Mean_ATE_PM_Match_PD_true,
                            std_ATE_PM_Match_PD_true))
        file1.write("\nUsing  NN PM PD(GAN), predicted ATE: {0}, SD: {1}"
                    .format(Mean_ATE_PM_Match_PD_predicted,
                            std_ATE_PM_Match_PD_predicted))
        file1.write("\n##################################################")

        file1.write("\n##################################################")
        file1.write("\n")
        file1.write("\nUsing NN No Dropout (PS Match), MSE: {0}, SD: {1}"
                    .format(MSE_mean_No_PM_Match_PD, std_MSE_No_PM_Match_PD))
        file1.write("\nUsing NN No Dropout (PS Match), true ATE: {0}, SD: {1}"
                    .format(Mean_ATE_No_PM_Match_PD_true,
                            std_ATE_No_PM_Match_PD_true))
        file1.write("\nUsing NN No Dropout (PS Match), predicted ATE: {0}, SD: {1}"
                    .format(Mean_ATE_No_PM_Match_PD_predicted,
                            std_ATE_No_PM_Match_PD_predicted))
        file1.write("\n##################################################")

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

            run_parameters["nn_DCN_PS_Match_No_PD_iter_file"] = "./MSE/ITE/ITE_DCN_PS_Match_No_PD_iter_{0}.csv"

            run_parameters["nn_DCN_PS_Match_PD_iter_file"] = "./MSE/ITE/ITE_DCN_Ps_Match_PD_iter_{0}.csv"
            run_parameters["nn_DCN_No_PS_Match_PD_iter_file"] = "./MSE/ITE/ITE_DCN_No_PS_Match_PD_iter_{0}.csv"

            run_parameters["summary_file_name"] = "Details_original.txt"
            run_parameters["is_synthetic"] = False

        elif running_mode == "synthetic_data":
            run_parameters["input_nodes"] = 75
            run_parameters["consolidated_file_path"] = "./MSE_Augmented/Results_consolidated.csv"

            run_parameters["is_synthetic"] = True

        return run_parameters

    @staticmethod
    def load_data(running_mode, dL, csv_path, split_size):
        if running_mode == "original_data":
            return dL.preprocess_data_from_csv(csv_path, split_size)

        elif running_mode == "synthetic_data":
            return dL.preprocess_data_from_csv_augmented(csv_path, split_size)
