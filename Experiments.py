from collections import OrderedDict
from datetime import date

import numpy as np

from Constants import Constants
# from PM_GAN import PM_GAN
from DCN_Experiments import DCN_Experiments
from Metrics import Metrics
from PS_Manager import PS_Manager
from Utils import Utils
from dataloader import DataLoader


class Experiments:
    def __init__(self, running_mode, csv_path, split_size):
        self.dL = DataLoader()
        self.ps_model = None
        self.running_mode = running_mode
        self.np_covariates_X_train, self.np_covariates_X_test, self.np_covariates_Y_train, \
        self.np_covariates_Y_test \
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
            print("--" * 20)
            input_nodes = run_parameters["input_nodes"]
            # get propensity score for classifier training and testing
            ps_score_list_train, ps_score_list_test = self.__get_ps_model(ps_model_type, iter_id,
                                                                          run_parameters["input_nodes"],
                                                                          device)
            data_loader_dict_train = self.dL.prepare_tensor_for_DCN(self.np_covariates_X_train,
                                                                    self.np_covariates_Y_train,
                                                                    ps_score_list_train,
                                                                    run_parameters["is_synthetic"])
            data_loader_dict_test = self.dL.prepare_tensor_for_DCN(self.np_covariates_X_test,
                                                                   self.np_covariates_Y_test,
                                                                   ps_score_list_test,
                                                                   run_parameters["is_synthetic"])
            # Execute PM GAN

            # run DCN Models
            dcn_experiments = DCN_Experiments(data_loader_dict_train, data_loader_dict_test,
                                              input_nodes, device)
            dcn_pd_models_eval_dict = dcn_experiments.evaluate_DCN_Model()
            dcn_pd_eval = dcn_pd_models_eval_dict["dcn_pd_eval_dict"]
            dcn_pd_PEHE, dcn_pd_ATE = self.process_evaluated_metric(dcn_pd_eval["y1_true_list"],
                                                                    dcn_pd_eval["y0_true_list"],
                                                                    dcn_pd_eval["y1_hat_list"],
                                                                    dcn_pd_eval["y0_hat_list"],
                                                                    dcn_pd_eval["ITE_dict_list"],
                                                                    run_parameters["DCN_PD"],
                                                                    iter_id)
            result_dict = OrderedDict()
            result_dict["iter_id"] = iter_id
            result_dict["PEHE_DCN_PD"] = dcn_pd_PEHE
            result_dict["ATE_Metric_DCN_PD"] = dcn_pd_ATE
            result_dict["true_ATE_DCN_PD"] = dcn_pd_eval["true_ITE"]
            result_dict["predicted_DCN_PD"] = dcn_pd_eval["predicted_ITE"]


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

    def __get_run_parameters(self):
        run_parameters = {}
        if self.running_mode == "original_data":
            run_parameters["input_nodes"] = 25
            run_parameters["consolidated_file_path"] = "./MSE/Results_consolidated.csv"

            # NN
            run_parameters["nn_prop_file"] = "./MSE/NN_Prop_score_{0}.csv"

            run_parameters["DCN_PD"] = "./MSE/ITE/ITE_DCN_PD_iter_{0}.csv"

            run_parameters["nn_DCN_PS_Match_PD_iter_file"] = "./MSE/ITE/ITE_DCN_Ps_Match_PD_iter_{0}.csv"
            run_parameters["nn_DCN_No_PS_Match_PD_iter_file"] = "./MSE/ITE/ITE_DCN_No_PS_Match_PD_iter_{0}.csv"

            run_parameters["summary_file_name"] = "Details_original.txt"
            run_parameters["is_synthetic"] = False

        elif self.running_mode == "synthetic_data":
            run_parameters["input_nodes"] = 75
            run_parameters["consolidated_file_path"] = "./MSE_Augmented/Results_consolidated.csv"

            run_parameters["is_synthetic"] = True

        return run_parameters

    def __load_data(self, csv_path, split_size):
        if self.running_mode == "original_data":
            return self.dL.preprocess_data_from_csv(csv_path, split_size)

        elif self.running_mode == "synthetic_data":
            return self.dL.preprocess_data_from_csv_augmented(csv_path, split_size)

    def __get_ps_model(self, ps_model_type, iter_id,
                       input_nodes, device):
        ps_train_set = self.dL.convert_to_tensor(self.np_covariates_X_train, self.np_covariates_Y_train)
        ps_test_set = self.dL.convert_to_tensor(self.np_covariates_X_test,
                                                self.np_covariates_Y_test)
        ps_manager = PS_Manager()
        if ps_model_type == Constants.PS_MODEL_NN:
            return ps_manager.get_propensity_scores(ps_train_set,
                                                    ps_test_set, iter_id,
                                                    input_nodes, device)

    def process_evaluated_metric(self, y1_true, y0_true, y1_hat, y0_hat,
                                 ite_dict, ite_csv_path, iter_id):
        y1_true_np = np.array(y1_true)
        y0_true_np = np.array(y0_true)
        y1_hat_np = np.array(y1_hat)
        y0_hat_np = np.array(y0_hat)

        PEHE = Metrics.PEHE(y1_true_np, y0_true_np, y1_hat_np, y0_hat_np)
        ATE = Metrics.ATE(y1_true_np, y0_true_np, y1_hat_np, y0_hat_np)
        print("PEHE: {0}".format(PEHE))
        print("ATE: {0}".format(ATE))

        Utils.write_to_csv(ite_csv_path.format(iter_id), ite_dict)
        return  PEHE, ATE
