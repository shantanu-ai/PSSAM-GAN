from collections import OrderedDict
from datetime import date

import numpy as np

from Constants import Constants
from DCN_Experiments import DCN_Experiments
from PS_Manager import PS_Manager
from PS_Treated_Generator import PS_Treated_Generator
from Utils import Utils
from dataloader import DataLoader


class Experiments:
    def __init__(self, running_mode):
        self.dL = DataLoader()
        self.running_mode = running_mode
        self.np_covariates_X_train = None
        self.np_covariates_X_test = None
        self.np_covariates_T_train = None
        self.np_covariates_T_test = None

    def run_all_experiments(self, train_path, test_path, iterations, ps_model_type):
        device = Utils.get_device()
        print(device)
        results_list = []

        run_parameters = self.__get_run_parameters()
        print(str(run_parameters["summary_file_name"]))
        file1 = open(run_parameters["summary_file_name"], "a")
        for iter_id in range(iterations):
            print("--" * 20)
            print("iter_id: {0}".format(iter_id))
            print("Jobs - NN")
            print("--" * 20)
            input_nodes = run_parameters["input_nodes"]
            self.np_covariates_X_train, self.np_covariates_X_test, self.np_covariates_T_train, \
            self.np_covariates_T_test \
                = self.__load_data(train_path, test_path, iter_id)
            # get propensity score for classifier training and testing
            ps_score_list_train, ps_score_list_test, ps_model = self.__get_ps_model(ps_model_type,
                                                                                    iter_id,
                                                                                    run_parameters["input_nodes"],
                                                                                    device)
            run_parameters["consolidated_file_path"] = self.get_consolidated_file_name(ps_model_type)

            print("--->>Train size: ")
            data_loader_dict_train = self.dL.prepare_tensor_for_DCN(self.np_covariates_X_train,
                                                                    self.np_covariates_T_train,
                                                                    ps_score_list_train,
                                                                    run_parameters["is_synthetic"])

            print("--->>Test size: ")
            data_loader_dict_test = self.dL.prepare_tensor_for_DCN(self.np_covariates_X_test,
                                                                   self.np_covariates_T_test,
                                                                   ps_score_list_test,
                                                                   run_parameters["is_synthetic"])

            n_treated_original = data_loader_dict_train["treated_data"][0].shape[0]
            n_control_original = data_loader_dict_train["control_data"][0].shape[0]

            # Execute PM GAN
            ps_t = PS_Treated_Generator(data_loader_dict_train, ps_model, ps_model_type)

            balanced_dataset_dict = ps_t.simulate_treated_semi_supervised(input_nodes, iter_id, device)
            tensor_treated_balanced_dcn = balanced_dataset_dict["tensor_treated_balanced_dcn"]
            tensor_control_balanced_dcn = balanced_dataset_dict["tensor_control_balanced_dcn"]
            n_treated_balanced_dcn = balanced_dataset_dict["n_treated_balanced_dcn"]
            n_control_balanced_dcn = balanced_dataset_dict["n_control_balanced_dcn"]
            # tensor_treated_balanced_tarnet = balanced_dataset_dict["tensor_treated_balanced_tarnet"]
            # tuple_control_balanced_tarnet = balanced_dataset_dict["tuple_control_balanced_tarnet"]

            print("---" * 20)
            print("-----------> !! Supervised Training(DCN Models ) !!<-----------")
            # run DCN Models
            tensor_treated_train_original = \
                Utils.create_tensors_from_tuple(data_loader_dict_train["treated_data"])
            tensor_control_train_original = \
                Utils.create_tensors_from_tuple(data_loader_dict_train["control_data"])

            model_save_paths = {
                "Model_DCN_PD_shared": run_parameters["Model_DCN_PD_shared"].format(iter_id),
                "Model_DCN_PD_y1": run_parameters["Model_DCN_PD_y1"].format(iter_id),
                "Model_DCN_PD_y0": run_parameters["Model_DCN_PD_y0"].format(iter_id),

                "Model_DCN_PD_02_shared": run_parameters["Model_DCN_PD_02_shared"].format(iter_id),
                "Model_DCN_PD_02_y1": run_parameters["Model_DCN_PD_02_y1"].format(iter_id),
                "Model_DCN_PD_02_y0": run_parameters["Model_DCN_PD_02_y0"].format(iter_id),

                "Model_DCN_PD_05_shared": run_parameters["Model_DCN_PD_05_shared"].format(iter_id),
                "Model_DCN_PD_05_y1": run_parameters["Model_DCN_PD_05_y1"].format(iter_id),
                "Model_DCN_PD_05_y0": run_parameters["Model_DCN_PD_05_y0"].format(iter_id),

                "Model_DCN_PM_GAN_shared": run_parameters["Model_DCN_PM_GAN_shared"].format(iter_id),
                "Model_DCN_PM_GAN_y1": run_parameters["Model_DCN_PM_GAN_y1"].format(iter_id),
                "Model_DCN_PM_GAN_y0": run_parameters["Model_DCN_PM_GAN_y0"].format(iter_id),

                "Model_DCN_PM_GAN_02_shared": run_parameters["Model_DCN_PM_GAN_02_shared"].format(iter_id),
                "Model_DCN_PM_GAN_02_y1": run_parameters["Model_DCN_PM_GAN_02_y1"].format(iter_id),
                "Model_DCN_PM_GAN_02_y0": run_parameters["Model_DCN_PM_GAN_02_y0"].format(iter_id),

                "Model_DCN_PM_GAN_05_shared": run_parameters["Model_DCN_PM_GAN_05_shared"].format(iter_id),
                "Model_DCN_PM_GAN_05_y1": run_parameters["Model_DCN_PM_GAN_05_y1"].format(iter_id),
                "Model_DCN_PM_GAN_05_y0": run_parameters["Model_DCN_PM_GAN_05_y0"].format(iter_id),

                "Model_DCN_PM_GAN_PD_shared": run_parameters["Model_DCN_PM_GAN_PD_shared"].format(iter_id),
                "Model_DCN_PM_GAN_PD_y1": run_parameters["Model_DCN_PM_GAN_PD_y1"].format(iter_id),
                "Model_DCN_PM_GAN_PD_y0": run_parameters["Model_DCN_PM_GAN_PD_y0"].format(iter_id)
            }

            dcn_experiments = DCN_Experiments(input_nodes, device)
            dcn_pd_models_eval_dict = dcn_experiments.evaluate_DCN_Model(tensor_treated_train_original,
                                                                         tensor_control_train_original,
                                                                         n_treated_original,
                                                                         n_control_original,
                                                                         tensor_treated_balanced_dcn,
                                                                         tensor_control_balanced_dcn,
                                                                         n_treated_balanced_dcn,
                                                                         n_control_balanced_dcn,
                                                                         data_loader_dict_test,
                                                                         model_save_paths)

            print("---" * 20)
            print("-----------> !! Supervised Evaluation(DCN Models) !! <-----------")
            print("---" * 20)
            print("--> 1. Model 1: DCN - PD Supervised Training Evaluation: ")
            dcn_pd_eval = dcn_pd_models_eval_dict["dcn_pd_eval_dict"]
            dcn_pd_ate_pred, dcn_pd_att_pred, dcn_pd_bias_att, dcn_pd_atc_pred, dcn_pd_policy_value, \
            dcn_pd_policy_risk, dcn_pd_err_fact = \
                self.__process_evaluated_metric(
                    dcn_pd_eval["yf_list"],
                    dcn_pd_eval["e_list"],
                    dcn_pd_eval["T_list"],
                    dcn_pd_eval["y1_hat_list"],
                    dcn_pd_eval["y0_hat_list"],
                    dcn_pd_eval["ITE_dict_list"],
                    dcn_pd_eval["predicted_ITE"],
                    run_parameters["DCN_PD"],
                    iter_id)
            print("---" * 20)

            print("--> 2. Model 2: DCN - PD(Dropout 0.2) Supervised Training Evaluation: ")
            dcn_pd_02_eval_dict = dcn_pd_models_eval_dict["dcn_pd_02_eval_dict"]
            dcn_pd_02_ate_pred, dcn_pd_02_att_pred, dcn_pd_02_bias_att, dcn_pd_02_atc_pred, \
            dcn_pd_02_policy_value, \
            dcn_pd_02_policy_risk, dcn_pd_02_err_fact = \
                self.__process_evaluated_metric(
                    dcn_pd_02_eval_dict["yf_list"],
                    dcn_pd_02_eval_dict["e_list"],
                    dcn_pd_02_eval_dict["T_list"],
                    dcn_pd_02_eval_dict["y1_hat_list"],
                    dcn_pd_02_eval_dict["y0_hat_list"],
                    dcn_pd_02_eval_dict["ITE_dict_list"],
                    dcn_pd_02_eval_dict["predicted_ITE"],
                    run_parameters["DCN_PD_02"],
                    iter_id)
            print("---" * 20)

            print("--> 3. Model 3: DCN - PD(Dropout 0.5) Supervised Training Evaluation: ")
            dcn_pd_05_eval_dict = dcn_pd_models_eval_dict["dcn_pd_05_eval_dict"]
            dcn_pd_05_ate_pred, dcn_pd_05_att_pred, dcn_pd_05_bias_att, dcn_pd_05_atc_pred, \
            dcn_pd_05_policy_value, \
            dcn_pd_05_policy_risk, dcn_pd_05_err_fact = \
                self.__process_evaluated_metric(
                    dcn_pd_05_eval_dict["yf_list"],
                    dcn_pd_05_eval_dict["e_list"],
                    dcn_pd_05_eval_dict["T_list"],
                    dcn_pd_05_eval_dict["y1_hat_list"],
                    dcn_pd_05_eval_dict["y0_hat_list"],
                    dcn_pd_05_eval_dict["ITE_dict_list"],
                    dcn_pd_05_eval_dict["predicted_ITE"],
                    run_parameters["DCN_PD_05"],
                    iter_id)
            print("---" * 20)

            print("--> 4. Model 4: PM GAN - No dropout Supervised Training Evaluation: ")
            dcn_pm_gan_eval = dcn_pd_models_eval_dict["dcn_pm_gan_eval_dict"]
            dcn_pm_gan_ate_pred, dcn_pm_gan_att_pred, dcn_pm_gan_bias_att, dcn_pm_gan_atc_pred, \
            dcn_pm_gan_policy_value, dcn_pm_gan_policy_risk, dcn_pm_gan_err_fact = \
                self.__process_evaluated_metric(
                    dcn_pm_gan_eval["yf_list"],
                    dcn_pm_gan_eval["e_list"],
                    dcn_pm_gan_eval["T_list"],
                    dcn_pm_gan_eval["y1_hat_list"],
                    dcn_pm_gan_eval["y0_hat_list"],
                    dcn_pm_gan_eval["ITE_dict_list"],
                    dcn_pm_gan_eval["predicted_ITE"],
                    run_parameters["DCN_PM_GAN"],
                    iter_id)
            print("---" * 20)

            print("--> 5. Model 5: PM GAN - dropout 0.2 Supervised Training Evaluation: ")
            dcn_pm_gan_eval_02 = dcn_pd_models_eval_dict["dcn_pm_gan_eval_drp_02_dict"]
            dcn_pm_gan_02_ate_pred, dcn_pm_gan_02_att_pred, dcn_pm_gan_02_bias_att, dcn_pm_gan_02_atc_pred, \
            dcn_pm_gan_02_policy_value, dcn_pm_gan_02_policy_risk, dcn_pm_gan_02_err_fact = \
                self.__process_evaluated_metric(
                    dcn_pm_gan_eval_02["yf_list"],
                    dcn_pm_gan_eval_02["e_list"],
                    dcn_pm_gan_eval_02["T_list"],
                    dcn_pm_gan_eval_02["y1_hat_list"],
                    dcn_pm_gan_eval_02["y0_hat_list"],
                    dcn_pm_gan_eval_02["ITE_dict_list"],
                    dcn_pm_gan_eval_02["predicted_ITE"],
                    run_parameters["DCN_PM_GAN_02"],
                    iter_id)
            print("---" * 20)

            print("--> 6. Model 6: PM GAN - dropout 0.5 Supervised Training Evaluation: ")
            dcn_pm_gan_eval_05 = dcn_pd_models_eval_dict["dcn_pm_gan_eval_drp_05_dict"]
            dcn_pm_gan_05_ate_pred, dcn_pm_gan_05_att_pred, dcn_pm_gan_05_bias_att, dcn_pm_gan_05_atc_pred, \
            dcn_pm_gan_05_policy_value, dcn_pm_gan_05_policy_risk, dcn_pm_gan_05_err_fact = \
                self.__process_evaluated_metric(
                    dcn_pm_gan_eval_05["yf_list"],
                    dcn_pm_gan_eval_05["e_list"],
                    dcn_pm_gan_eval_05["T_list"],
                    dcn_pm_gan_eval_05["y1_hat_list"],
                    dcn_pm_gan_eval_05["y0_hat_list"],
                    dcn_pm_gan_eval_05["ITE_dict_list"],
                    dcn_pm_gan_eval_05["predicted_ITE"],
                    run_parameters["DCN_PM_GAN_05"],
                    iter_id)
            print("---" * 20)

            print("--> 7. Model 7: PM GAN - PD Supervised Training Evaluation: ")
            dcn_pm_gan_eval_pd = dcn_pd_models_eval_dict["dcn_pm_gan_eval_pd_dict"]
            dcn_pm_gan_pd_ate_pred, dcn_pm_gan_pd_att_pred, dcn_pm_gan_pd_bias_att, dcn_pm_gan_pd_atc_pred, \
            dcn_pm_gan_pd_policy_value, dcn_pm_gan_pd_policy_risk, dcn_pm_gan_pd_err_fact = \
                self.__process_evaluated_metric(
                    dcn_pm_gan_eval_pd["yf_list"],
                    dcn_pm_gan_eval_pd["e_list"],
                    dcn_pm_gan_eval_pd["T_list"],
                    dcn_pm_gan_eval_pd["y1_hat_list"],
                    dcn_pm_gan_eval_pd["y0_hat_list"],
                    dcn_pm_gan_eval_pd["ITE_dict_list"],
                    dcn_pm_gan_eval_pd["predicted_ITE"],
                    run_parameters["DCN_PM_GAN_PD"],
                    iter_id)
            print("---" * 20)

            print("---" * 20)

            # run TARNet Models
            tarnet_PEHE = 0
            tarnet_pm_gan_PEHE = 0
            # print("-----------> !! Supervised Training(TARNet Models) !!<-----------")
            # tarnet_experiments = TARNet_Experiments(input_nodes, device)
            # tarnet_experiments_models_eval_dict = tarnet_experiments.evaluate_TARNet_Model(
            #     data_loader_dict_test["treated_data"],
            #     data_loader_dict_test["control_data"],
            #     tensor_treated_balanced_tarnet,
            #     tuple_control_balanced_tarnet,
            #     data_loader_dict_test)
            #
            # tarnet_eval = tarnet_experiments_models_eval_dict["tarnet_eval_dict"]
            # print("---" * 20)
            # print("---> !! Supervised Evaluation(TARNet Models) !! <---")
            # print("---" * 20)
            # print("--> 1. Model 1: TARNet Supervised Training Evaluation: ")
            # tarnet_PEHE, tarnet_ATE_metric, tarnet_true_ATE, tarnet_predicted_ATE = \
            #     self.__process_evaluated_metric(
            #         tarnet_eval["y1_true_list"],
            #         tarnet_eval["y0_true_list"],
            #         tarnet_eval["y1_hat_list"],
            #         tarnet_eval["y0_hat_list"],
            #         tarnet_eval["ITE_dict_list"],
            #         tarnet_eval["true_ITE"],
            #         tarnet_eval["predicted_ITE"],
            #         run_parameters["TARNET"],
            #         iter_id)
            #
            # tarnet_pm_gan_eval = tarnet_experiments_models_eval_dict["tarnet_pm_gan_eval_dict"]
            # print("--> 2. Model 2: TARNet PM GAN Supervised Training Evaluation: ")
            # tarnet_pm_gan_PEHE, tarnet_pm_gan_ATE_metric, tarnet_pm_gan_true_ATE, tarnet_pm_gan_predicted_ATE = \
            #     self.__process_evaluated_metric(
            #         tarnet_pm_gan_eval["y1_true_list"],
            #         tarnet_pm_gan_eval["y0_true_list"],
            #         tarnet_pm_gan_eval["y1_hat_list"],
            #         tarnet_pm_gan_eval["y0_hat_list"],
            #         tarnet_pm_gan_eval["ITE_dict_list"],
            #         tarnet_pm_gan_eval["true_ITE"],
            #         tarnet_pm_gan_eval["predicted_ITE"],
            #         run_parameters["TARNET_PM_GAN"],
            #         iter_id)
            #
            # print("---" * 20)

            result_dict = OrderedDict()
            result_dict["iter_id"] = iter_id
            result_dict["dcn_pd_ate_pred"] = dcn_pd_ate_pred
            result_dict["dcn_pd_att_pred"] = dcn_pd_att_pred
            result_dict["dcn_pd_bias_att"] = dcn_pd_bias_att
            result_dict["dcn_pd_atc_pred"] = dcn_pd_atc_pred
            result_dict["dcn_pd_policy_value"] = dcn_pd_policy_value
            result_dict["dcn_pd_policy_risk"] = dcn_pd_policy_risk
            result_dict["dcn_pd_err_fact"] = dcn_pd_err_fact

            result_dict["dcn_pd_02_ate_pred"] = dcn_pd_02_ate_pred
            result_dict["dcn_pd_02_att_pred"] = dcn_pd_02_att_pred
            result_dict["dcn_pd_02_bias_att"] = dcn_pd_02_bias_att
            result_dict["dcn_pd_02_atc_pred"] = dcn_pd_02_atc_pred
            result_dict["dcn_pd_02_policy_value"] = dcn_pd_02_policy_value
            result_dict["dcn_pd_02_policy_risk"] = dcn_pd_02_policy_risk
            result_dict["dcn_pd_02_err_fact"] = dcn_pd_02_err_fact

            result_dict["dcn_pd_05_ate_pred"] = dcn_pd_05_ate_pred
            result_dict["dcn_pd_05_att_pred"] = dcn_pd_05_att_pred
            result_dict["dcn_pd_05_bias_att"] = dcn_pd_05_bias_att
            result_dict["dcn_pd_05_atc_pred"] = dcn_pd_05_atc_pred
            result_dict["dcn_pd_05_policy_value"] = dcn_pd_05_policy_value
            result_dict["dcn_pd_05_policy_risk"] = dcn_pd_05_policy_risk
            result_dict["dcn_pd_05_err_fact"] = dcn_pd_05_err_fact

            result_dict["dcn_pm_gan_ate_pred"] = dcn_pm_gan_ate_pred
            result_dict["dcn_pm_gan_att_pred"] = dcn_pm_gan_att_pred
            result_dict["dcn_pm_gan_bias_att"] = dcn_pm_gan_bias_att
            result_dict["dcn_pm_gan_atc_pred"] = dcn_pm_gan_atc_pred
            result_dict["dcn_pm_gan_policy_value"] = dcn_pm_gan_policy_value
            result_dict["dcn_pm_gan_policy_risk"] = dcn_pm_gan_policy_risk
            result_dict["dcn_pm_gan_err_fact"] = dcn_pm_gan_err_fact

            result_dict["dcn_pm_gan_02_ate_pred"] = dcn_pm_gan_02_ate_pred
            result_dict["dcn_pm_gan_02_att_pred"] = dcn_pm_gan_02_att_pred
            result_dict["dcn_pm_gan_02_bias_att"] = dcn_pm_gan_02_bias_att
            result_dict["dcn_pm_gan_02_atc_pred"] = dcn_pm_gan_02_atc_pred
            result_dict["dcn_pm_gan_02_policy_value"] = dcn_pm_gan_02_policy_value
            result_dict["dcn_pm_gan_02_policy_risk"] = dcn_pm_gan_02_policy_risk
            result_dict["dcn_pm_gan_02_err_fact"] = dcn_pm_gan_02_err_fact

            result_dict["dcn_pm_gan_05_att_pred"] = dcn_pm_gan_05_ate_pred
            result_dict["dcn_pm_gan_05_att_pred"] = dcn_pm_gan_05_att_pred
            result_dict["dcn_pm_gan_05_bias_att"] = dcn_pm_gan_05_bias_att
            result_dict["dcn_pm_gan_05_atc_pred"] = dcn_pm_gan_05_atc_pred
            result_dict["dcn_pm_gan_05_policy_value"] = dcn_pm_gan_05_policy_value
            result_dict["dcn_pm_gan_05_policy_risk"] = dcn_pm_gan_05_policy_risk
            result_dict["dcn_pm_gan_05_err_fact"] = dcn_pm_gan_05_err_fact

            result_dict["dcn_pm_gan_pd_att_pred"] = dcn_pm_gan_pd_ate_pred
            result_dict["dcn_pm_gan_pd_att_pred"] = dcn_pm_gan_pd_att_pred
            result_dict["dcn_pm_gan_pd_bias_att"] = dcn_pm_gan_pd_bias_att
            result_dict["dcn_pm_gan_pd_atc_pred"] = dcn_pm_gan_pd_atc_pred
            result_dict["dcn_pm_gan_pd_policy_value"] = dcn_pm_gan_pd_policy_value
            result_dict["dcn_pm_gan_pd_policy_risk"] = dcn_pm_gan_pd_policy_risk
            result_dict["dcn_pm_gan_pd_err_fact"] = dcn_pm_gan_pd_err_fact

            # result_dict["tarnet_PEHE"] = tarnet_PEHE
            # result_dict["tarnet_ATE_metric"] = tarnet_ATE_metric
            # result_dict["tarnet_true_ATE"] = tarnet_true_ATE
            # result_dict["tarnet_predicted_ATE"] = tarnet_predicted_ATE
            #
            # result_dict["tarnet_pm_gan_PEHE"] = tarnet_pm_gan_PEHE
            # result_dict["tarnet_pm_gan_ATE_metric"] = tarnet_pm_gan_ATE_metric
            # result_dict["tarnet_pm_gan_true_ATE"] = tarnet_pm_gan_true_ATE
            # result_dict["tarnet_pm_gan_predicted_ATE"] = tarnet_pm_gan_predicted_ATE

            file1.write("\nToday's date: {0}\n".format(date.today()))
            file1.write("Iter: {0}, bias_att_DCN_PD: {1}, bias_att_DCN_PD(0.2): {2},  bias_att_DCN_PD(0.5): {3},  "
                        "bias_att_DCN_PM_GAN: {4},  "
                        "bias_att_DCN_PM_GAN_02: {5}, bias_att_DCN_PM_GAN_05: {6}, bias_att_DCN_PM_GAN(PD): {7}, "
                        "policy_risk_DCN_PD: {8},  policy_risk_DCN_PD(0.2): {9} , "
                        "policy_risk_DCN_PD(0.5): {10}, policy_risk_DCN_PM_GAN: {11},  "
                        "policy_risk_DCN_PM_GAN_02: {12}, policy_risk_PM_GAN_05: {13}, policy_risk_PM_GAN(PD): {14}, "

                        .format(iter_id, dcn_pd_bias_att, dcn_pd_02_bias_att,
                                dcn_pd_05_bias_att,
                                dcn_pm_gan_bias_att,
                                dcn_pm_gan_02_bias_att, dcn_pm_gan_05_bias_att, dcn_pm_gan_pd_bias_att,
                                dcn_pd_policy_risk, dcn_pd_02_policy_risk, dcn_pd_05_policy_risk,
                                dcn_pm_gan_policy_risk,
                                dcn_pm_gan_02_policy_risk, dcn_pm_gan_05_policy_risk,
                                dcn_pm_gan_pd_policy_risk))
            results_list.append(result_dict)

        bias_att_set_DCN_PD = []
        policy_risk_set_DCN_PD = []

        bias_att_set_DCN_PD_02 = []
        policy_risk_set_DCN_PD_02 = []

        bias_att_set_DCN_PD_05 = []
        policy_risk_set_DCN_PD_05 = []

        bias_att_DCN_PM_GAN = []
        policy_risk_set_DCN_PM_GAN = []

        bias_att_DCN_PM_GAN_02 = []
        policy_risk_set_DCN_PM_GAN_02 = []

        bias_att_DCN_PM_GAN_05 = []
        policy_risk_set_DCN_PM_GAN_05 = []

        bias_att_DCN_PM_GAN_PD = []
        policy_risk_set_DCN_PM_GAN_PD = []

        # PEHE_set_Tarnet = []
        # ATE_Metric_set_Tarnet = []
        # true_ATE_set_Tarnet = []
        # predicted_ATE_Tarnet = []
        #
        # PEHE_set_Tarnet_PM_GAN = []
        # ATE_Metric_set_Tarnet_PM_GAN = []
        # true_ATE_set_Tarnet_PM_GAN = []
        # predicted_ATE_Tarnet_PM_GAN = []

        for result in results_list:
            bias_att_set_DCN_PD.append(result["dcn_pd_bias_att"])
            policy_risk_set_DCN_PD.append(result["dcn_pd_policy_risk"])

            bias_att_set_DCN_PD_02.append(result["dcn_pd_02_bias_att"])
            policy_risk_set_DCN_PD_02.append(result["dcn_pd_02_policy_risk"])

            bias_att_set_DCN_PD_05.append(result["dcn_pd_05_bias_att"])
            policy_risk_set_DCN_PD_05.append(result["dcn_pd_05_policy_risk"])

            bias_att_DCN_PM_GAN.append(result["dcn_pm_gan_bias_att"])
            policy_risk_set_DCN_PM_GAN.append(result["dcn_pm_gan_policy_risk"])

            bias_att_DCN_PM_GAN_02.append(result["dcn_pm_gan_02_bias_att"])
            policy_risk_set_DCN_PM_GAN_02.append(result["dcn_pm_gan_02_policy_risk"])

            bias_att_DCN_PM_GAN_05.append(result["dcn_pm_gan_05_bias_att"])
            policy_risk_set_DCN_PM_GAN_05.append(result["dcn_pm_gan_05_policy_risk"])

            bias_att_DCN_PM_GAN_PD.append(result["dcn_pm_gan_pd_bias_att"])
            policy_risk_set_DCN_PM_GAN_PD.append(result["dcn_pm_gan_pd_policy_risk"])

            # PEHE_set_Tarnet.append(result["tarnet_PEHE"])
            # ATE_Metric_set_Tarnet.append(result["tarnet_ATE_metric"])
            # true_ATE_set_Tarnet.append(result["tarnet_true_ATE"])
            # predicted_ATE_Tarnet.append(result["tarnet_predicted_ATE"])
            #
            # PEHE_set_Tarnet_PM_GAN.append(result["tarnet_pm_gan_PEHE"])
            # ATE_Metric_set_Tarnet_PM_GAN.append(result["tarnet_pm_gan_ATE_metric"])
            # true_ATE_set_Tarnet_PM_GAN.append(result["tarnet_pm_gan_true_ATE"])
            # predicted_ATE_Tarnet_PM_GAN.append(result["tarnet_pm_gan_predicted_ATE"])

        bias_att_DCN_PD_mean = np.mean(np.array(bias_att_set_DCN_PD))
        bias_att_DCN_PD_std = np.std(bias_att_set_DCN_PD)
        policy_risk_set_DCN_PD_mean = np.mean(np.array(policy_risk_set_DCN_PD))
        policy_risk_set_DCN_PD_std = np.std(policy_risk_set_DCN_PD)

        bias_att_DCN_PD_mean_02 = np.mean(np.array(bias_att_set_DCN_PD_02))
        bias_att_DCN_PD_std_02 = np.std(bias_att_set_DCN_PD_02)
        policy_risk_set_DCN_PD_mean_02 = np.mean(np.array(policy_risk_set_DCN_PD_02))
        policy_risk_set_DCN_PD_std_02 = np.std(policy_risk_set_DCN_PD_02)

        bias_att_DCN_PD_mean_05 = np.mean(np.array(bias_att_set_DCN_PD_05))
        bias_att_DCN_PD_std_05 = np.std(bias_att_set_DCN_PD_05)
        policy_risk_set_DCN_PD_mean_05 = np.mean(np.array(policy_risk_set_DCN_PD_05))
        policy_risk_set_DCN_PD_std_05 = np.std(policy_risk_set_DCN_PD_05)

        bias_att_DCN_PM_GAN_mean = np.mean(np.array(bias_att_DCN_PM_GAN))
        bias_att_DCN_PM_GAN_std = np.std(bias_att_DCN_PM_GAN)
        policy_risk_set_DCN_PM_GAN_mean = np.mean(np.array(policy_risk_set_DCN_PM_GAN))
        policy_risk_set_DCN_PM_GAN_std = np.std(policy_risk_set_DCN_PM_GAN)

        bias_att_DCN_PM_GAN_02_mean = np.mean(np.array(bias_att_DCN_PM_GAN_02))
        bias_att_DCN_PM_GAN_02_std = np.std(bias_att_DCN_PM_GAN_02)
        policy_risk_DCN_PM_GAN_02_mean = np.mean(np.array(policy_risk_set_DCN_PM_GAN_02))
        policy_risk_DCN_PM_GAN_02_std = np.std(policy_risk_set_DCN_PM_GAN_02)

        bias_att_DCN_PM_GAN_05_mean = np.mean(np.array(bias_att_DCN_PM_GAN_05))
        bias_att_DCN_PM_GAN_05_std = np.std(bias_att_DCN_PM_GAN_05)
        policy_risk_DCN_PM_GAN_05_mean = np.mean(np.array(policy_risk_set_DCN_PM_GAN_05))
        policy_risk_DCN_PM_GAN_05_std = np.std(policy_risk_set_DCN_PM_GAN_05)

        bias_att_DCN_PM_GAN_mean_PD = np.mean(np.array(bias_att_DCN_PM_GAN_PD))
        bias_att_DCN_PM_GAN_std_PD = np.std(bias_att_DCN_PM_GAN_PD)
        policy_risk_DCN_PM_GAN_mean_PD = np.mean(np.array(policy_risk_set_DCN_PM_GAN_PD))
        policy_risk_DCN_PM_GAN_std_PD = np.std(policy_risk_set_DCN_PM_GAN_PD)

        # PEHE_set_Tarnet_mean = np.mean(np.array(PEHE_set_Tarnet))
        # PEHE_set_Tarnet_std = np.std(PEHE_set_Tarnet)
        # ATE_Metric_set_Tarnet_mean = np.mean(np.array(ATE_Metric_set_Tarnet))
        # ATE_Metric_set_Tarnet_std = np.std(ATE_Metric_set_Tarnet)
        # true_ATE_set_Tarnet_mean = np.mean(np.array(true_ATE_set_Tarnet))
        # true_ATE_set_Tarnet_std = np.std(true_ATE_set_Tarnet)
        # predicted_ATE_Tarnet_mean = np.mean(np.array(predicted_ATE_Tarnet))
        # predicted_ATE_Tarnet_std = np.std(predicted_ATE_Tarnet)
        #
        # PEHE_set_Tarnet_PM_GAN_mean = np.mean(np.array(PEHE_set_Tarnet_PM_GAN))
        # PEHE_set_Tarnet_PM_GAN_std = np.std(PEHE_set_Tarnet_PM_GAN)
        # ATE_Metric_set_Tarnet_PM_GAN_mean = np.mean(np.array(ATE_Metric_set_Tarnet_PM_GAN))
        # ATE_Metric_set_Tarnet_PM_GAN_std = np.std(ATE_Metric_set_Tarnet_PM_GAN)
        # true_ATE_set_Tarnet_PM_GAN_mean = np.mean(np.array(true_ATE_set_Tarnet_PM_GAN))
        # true_ATE_set_Tarnet_PM_GAN_std = np.std(true_ATE_set_Tarnet_PM_GAN)
        # predicted_ATE_Tarnet_PM_GAN_mean = np.mean(np.array(predicted_ATE_Tarnet_PM_GAN))
        # predicted_ATE_Tarnet_PM_GAN_std = np.std(predicted_ATE_Tarnet_PM_GAN)

        print("###" * 20)
        print("----------------- !!DCN Models(Results) !! ------------------------")
        print("--" * 20)

        print("Model 1: DCN_PD")
        print("DCN_PD, Bias: {0}, SD: {1}"
              .format(bias_att_DCN_PD_mean, bias_att_DCN_PD_std))
        print("DCN_PD, Policy Risk: {0}, SD: {1}"
              .format(policy_risk_set_DCN_PD_mean, policy_risk_set_DCN_PD_std))
        print("--" * 20)

        print("Model 2: DCN_PD(0.2)")
        print("DCN_PD(0.2), Bias: {0}, SD: {1}"
              .format(bias_att_DCN_PD_mean_02, bias_att_DCN_PD_std_02))
        print("DCN_PD(0.2), Policy Risk: {0}, SD: {1}"
              .format(policy_risk_set_DCN_PD_mean_02, policy_risk_set_DCN_PD_std_02))
        print("--" * 20)

        print("Model 3: DCN_PD(0.5)")
        print("DCN_PD(0.5), Bias: {0}, SD: {1}"
              .format(bias_att_DCN_PD_mean_05, bias_att_DCN_PD_std_05))
        print("DCN_PD(0.5), Policy Risk: {0}, SD: {1}"
              .format(policy_risk_set_DCN_PD_mean_05, policy_risk_set_DCN_PD_std_05))
        print("--" * 20)

        print("Model 4: DCN PM GAN")
        print("DCN PM GAN, Bias: {0}, SD: {1}"
              .format(bias_att_DCN_PM_GAN_mean, bias_att_DCN_PM_GAN_std))
        print("DCN PM GAN, Policy Risk: {0}, SD: {1}"
              .format(policy_risk_set_DCN_PM_GAN_mean, policy_risk_set_DCN_PM_GAN_std))
        print("--" * 20)

        print("Model 5: DCN PM GAN Dropout 0.2")
        print("DCN PM GAN Dropout 0.2, Bias: {0}, SD: {1}"
              .format(bias_att_DCN_PM_GAN_02_mean, bias_att_DCN_PM_GAN_02_std))
        print("DCN PM GAN Dropout 0.2, Policy Risk: {0}, SD: {1}"
              .format(policy_risk_DCN_PM_GAN_02_mean, policy_risk_DCN_PM_GAN_02_std))
        print("--" * 20)

        print("Model 6: DCN PM GAN Dropout 0.5")
        print("DCN PM GAN Dropout 0.5, Bias: {0}, SD: {1}"
              .format(bias_att_DCN_PM_GAN_05_mean, bias_att_DCN_PM_GAN_05_std))
        print("DCN PM GAN Dropout 0.5, Policy Risk: {0}, SD: {1}"
              .format(policy_risk_DCN_PM_GAN_05_mean, policy_risk_DCN_PM_GAN_05_std))
        print("--" * 20)


        print("Model 7: DCN PM GAN(PD)")
        print("DCN PM GAN(PD), Bias: {0}, SD: {1}"
              .format(bias_att_DCN_PM_GAN_mean_PD, bias_att_DCN_PM_GAN_std_PD))
        print("DCN PM GAN(PD), Policy Risk: {0}, SD: {1}"
              .format(policy_risk_DCN_PM_GAN_mean_PD, policy_risk_DCN_PM_GAN_std_PD))
        print("--" * 20)
        print("###" * 20)
        # print("----------------- !!TARNet Models(Results) !! ------------------------")
        # print("--" * 20)
        #
        # print("Model 1: TARNET")
        # print("TARNET, PEHE: {0}, SD: {1}"
        #       .format(PEHE_set_Tarnet_mean, PEHE_set_Tarnet_std))
        # print("TARNET, ATE Metric: {0}, SD: {1}"
        #       .format(ATE_Metric_set_Tarnet_mean, ATE_Metric_set_Tarnet_std))
        # print("TARNET, True ATE: {0}, SD: {1}"
        #       .format(true_ATE_set_Tarnet_mean, true_ATE_set_Tarnet_std))
        # print("TARNET, predicted ATE: {0}, SD: {1}"
        #       .format(predicted_ATE_Tarnet_mean,
        #               predicted_ATE_Tarnet_std))
        # print("--" * 20)
        #
        # print("Model 2: TARNET PM GAN")
        # print("TARNET PM GAN, PEHE: {0}, SD: {1}"
        #       .format(PEHE_set_Tarnet_PM_GAN_mean, PEHE_set_Tarnet_PM_GAN_std))
        # print("TARNET PM GAN, ATE Metric: {0}, SD: {1}"
        #       .format(ATE_Metric_set_Tarnet_PM_GAN_mean, ATE_Metric_set_Tarnet_PM_GAN_std))
        # print("TARNET PM GAN, True ATE: {0}, SD: {1}"
        #       .format(true_ATE_set_Tarnet_PM_GAN_mean, true_ATE_set_Tarnet_PM_GAN_std))
        # print("TARNET PM GAN, predicted ATE: {0}, SD: {1}"
        #       .format(predicted_ATE_Tarnet_PM_GAN_mean,
        #               predicted_ATE_Tarnet_PM_GAN_std))
        print("--" * 3)
        print("###" * 3)

        file1.write("\n###" * 3)
        file1.write("\nDCN Models")
        file1.write("\n--" * 3)
        file1.write("\nModel 1: DCN_PD")
        file1.write("\nDCN_PD, Bias att: {0}, SD: {1}"
                    .format(bias_att_DCN_PD_mean, bias_att_DCN_PD_std))
        file1.write("\nDCN_PD, Policy Risk: {0}, SD: {1}"
                    .format(policy_risk_set_DCN_PD_mean,
                            policy_risk_set_DCN_PD_std))

        file1.write("\n--" * 3)
        file1.write("\nModel 2: DCN_PD(0.2)")
        file1.write("\nDCN_PD(0.2, Bias att: {0}, SD: {1}"
                    .format(bias_att_DCN_PD_mean_02, bias_att_DCN_PD_std_02))
        file1.write("\nDCN_PD(0.2, Policy Risk: {0}, SD: {1}"
                    .format(policy_risk_set_DCN_PD_mean_02,
                            policy_risk_set_DCN_PD_std_02))

        file1.write("\n--" * 3)
        file1.write("\nModel 3: DCN_PD(0.5)")
        file1.write("\nDCN_PD(0.5), Bias att: {0}, SD: {1}"
                    .format(bias_att_DCN_PD_mean_05, bias_att_DCN_PD_std_05))
        file1.write("\nDCN_PD(0.5), Policy Risk: {0}, SD: {1}"
                    .format(policy_risk_set_DCN_PD_mean_05,
                            policy_risk_set_DCN_PD_std_05))

        file1.write("\n--" * 3)
        file1.write("\nModel 4: DCN PM GAN")
        file1.write("\nDCN PM GAN, Bias att: {0}, SD: {1}"
                    .format(bias_att_DCN_PM_GAN_mean, bias_att_DCN_PM_GAN_std))
        file1.write("\nDCN PM GAN, Policy Risk: {0}, SD: {1}"
                    .format(policy_risk_set_DCN_PM_GAN_mean,
                            policy_risk_set_DCN_PM_GAN_std))
        file1.write("\n--" * 3)

        file1.write("\nModel 5: DCN PM GAN Dropout 0.2")
        file1.write("\nDCN PM GAN Dropout 0.2, Bias att: {0}, SD: {1}"
                    .format(bias_att_DCN_PM_GAN_02_std, bias_att_DCN_PM_GAN_02_std))
        file1.write("\nDCN PM GAN Dropout 0.2, Policy Risk: {0}, SD: {1}"
                    .format(policy_risk_DCN_PM_GAN_02_mean,
                            policy_risk_DCN_PM_GAN_02_std))
        file1.write("\n--" * 3)

        file1.write("\nModel 6: DCN PM GAN Dropout 0.5")
        file1.write("\nDCN PM GAN Dropout 0.5, Bias att: {0}, SD: {1}"
                    .format(bias_att_DCN_PM_GAN_05_mean, bias_att_DCN_PM_GAN_05_std))
        file1.write("\nDCN PM GAN Dropout 0.5, Policy Risk: {0}, SD: {1}"
                    .format(policy_risk_DCN_PM_GAN_05_mean,
                            policy_risk_DCN_PM_GAN_05_std))

        file1.write("\n--" * 3)
        file1.write("\nModel 7: DCN PM GAN PD")
        file1.write("\nDCN PM GAN Dropout PD, Bias att: {0}, SD: {1}"
                    .format(bias_att_DCN_PM_GAN_mean_PD, bias_att_DCN_PM_GAN_std_PD))
        file1.write("\nDCN PM GAN Dropout PD, Policy Risk: {0}, SD: {1}"
                    .format(policy_risk_DCN_PM_GAN_mean_PD,
                            policy_risk_DCN_PM_GAN_std_PD))
        file1.write("\n--" * 3)
        file1.write("\n###" * 3)

        # file1.write("\nTARNET Models")
        file1.write("\n--" * 3)
        # file1.write("\nModel 1: TARNET")
        # file1.write("\nTARNET, PEHE: {0}, SD: {1}"
        #             .format(PEHE_set_Tarnet_mean, PEHE_set_Tarnet_std))
        # file1.write("\nTARNET, ATE Metric: {0}, SD: {1}"
        #             .format(ATE_Metric_set_Tarnet_mean,
        #                     ATE_Metric_set_Tarnet_std))
        # file1.write("\nTARNET, True ATE: {0}, SD: {1}"
        #             .format(true_ATE_set_Tarnet_mean,
        #                     true_ATE_set_Tarnet_std))
        # file1.write("\nTARNET, predicted ATE: {0}, SD: {1}"
        #             .format(predicted_ATE_Tarnet_mean,
        #                     predicted_ATE_Tarnet_std))
        # file1.write("\n--" * 20)
        # file1.write("\nModel 2: TARNET PM GAN")
        # file1.write("\nTARNET PM GAN, PEHE: {0}, SD: {1}"
        #             .format(PEHE_set_Tarnet_PM_GAN_mean, PEHE_set_Tarnet_PM_GAN_std))
        # file1.write("\nTARNET PM GAN, ATE Metric: {0}, SD: {1}"
        #             .format(ATE_Metric_set_Tarnet_PM_GAN_mean,
        #                     ATE_Metric_set_Tarnet_PM_GAN_std))
        # file1.write("\nTARNET PM GAN, True ATE: {0}, SD: {1}"
        #             .format(true_ATE_set_Tarnet_PM_GAN_mean,
        #                     true_ATE_set_Tarnet_PM_GAN_std))
        # file1.write("\nTARNET PM GAN, predicted ATE: {0}, SD: {1}"
        #             .format(predicted_ATE_Tarnet_PM_GAN_mean,
        #                     predicted_ATE_Tarnet_PM_GAN_std))
        file1.write("\n--" * 3)
        file1.write("\n###" * 3)

        Utils.write_to_csv(run_parameters["consolidated_file_path"], results_list)

    def __get_run_parameters(self):
        run_parameters = {}
        if self.running_mode == "original_data":
            run_parameters["input_nodes"] = 17
            # run_parameters["consolidated_file_path"] = "./MSE/Results_consolidated.csv"

            # NN
            run_parameters["nn_prop_file"] = "./MSE/NN_Prop_score_{0}.csv"

            # ite files DCN
            run_parameters["DCN_PD"] = "./MSE/ITE/ITE_DCN_PD_iter_{0}.csv"
            run_parameters["DCN_PD_02"] = "./MSE/ITE/ITE_DCN_PD_02_iter_{0}.csv"
            run_parameters["DCN_PD_05"] = "./MSE/ITE/ITE_DCN_PD_05_iter_{0}.csv"

            run_parameters["DCN_PM_GAN"] = "./MSE/ITE/ITE_DCN_PM_GAN_iter_{0}.csv"
            run_parameters["DCN_PM_GAN_02"] = "./MSE/ITE/ITE_DCN_PM_GAN_dropout_02_iter_{0}.csv"
            run_parameters["DCN_PM_GAN_05"] = "./MSE/ITE/ITE_DCN_PM_GAN_dropout_05_iter_{0}.csv"
            run_parameters["DCN_PM_GAN_PD"] = "./MSE/ITE/ITE_DCN_PM_GAN_dropout_PD_iter_{0}.csv"

            # model paths DCN
            run_parameters["Model_DCN_PD_shared"] = "./Models/DCN_PD/DCN_PD_shared_iter_{0}.pth"
            run_parameters["Model_DCN_PD_y1"] = "./Models/DCN_PD/DCN_PD_y1_iter_{0}.pth"
            run_parameters["Model_DCN_PD_y0"] = "./Models/DCN_PD/DCN_PD_y2_iter_{0}.pth"

            run_parameters["Model_DCN_PD_02_shared"] = "./Models/DCN_PD_02/DCN_PD_02_shared_iter_{0}.pth"
            run_parameters["Model_DCN_PD_02_y1"] = "./Models/DCN_PD_02/DCN_PD_02_y1_iter_{0}.pth"
            run_parameters["Model_DCN_PD_02_y0"] = "./Models/DCN_PD_02/DCN_PD_02_y2_iter_{0}.pth"

            run_parameters["Model_DCN_PD_05_shared"] = "./Models/DCN_PD_05/DCN_PD_05_shared_iter_{0}.pth"
            run_parameters["Model_DCN_PD_05_y1"] = "./Models/DCN_PD_05/DCN_PD_05_y1_iter_{0}.pth"
            run_parameters["Model_DCN_PD_05_y0"] = "./Models/DCN_PD_05/DCN_PD_05_y2_iter_{0}.pth"

            run_parameters["Model_DCN_PM_GAN_shared"] = "./Models/PM_GAN/DCN_PM_GAN_shared_iter_{0}.pth"
            run_parameters["Model_DCN_PM_GAN_y1"] = "./Models/PM_GAN/DCN_PM_GAN_iter_y1_{0}.pth"
            run_parameters["Model_DCN_PM_GAN_y0"] = "./Models/PM_GAN/DCN_PM_GAN_iter_y0_{0}.pth"

            run_parameters[
                "Model_DCN_PM_GAN_02_shared"] = "./Models/PM_GAN_DR_02/DCN_PM_GAN_dropout_02_shared_iter_{0}.pth"
            run_parameters["Model_DCN_PM_GAN_02_y1"] = "./Models/PM_GAN_DR_02/DCN_PM_GAN_dropout_02_y1_iter_{0}.pth"
            run_parameters["Model_DCN_PM_GAN_02_y0"] = "./Models/PM_GAN_DR_02/DCN_PM_GAN_dropout_02_y0_iter_{0}.pth"

            run_parameters[
                "Model_DCN_PM_GAN_05_shared"] = "./Models/PM_GAN_DR_05/DCN_PM_GAN_dropout_05_shared_iter_{0}.pth"
            run_parameters["Model_DCN_PM_GAN_05_y1"] = "./Models/PM_GAN_DR_05/DCN_PM_GAN_dropout_05_y1_iter_{0}.pth"
            run_parameters["Model_DCN_PM_GAN_05_y0"] = "./Models/PM_GAN_DR_05/DCN_PM_GAN_dropout_05_y0_iter_{0}.pth"

            run_parameters[
                "Model_DCN_PM_GAN_PD_shared"] = "./Models/PM_GAN_PD/DCN_PM_GAN_dropout_PD_shared_iter_{0}.pth"
            run_parameters["Model_DCN_PM_GAN_PD_y1"] = "./Models/PM_GAN_PD/DCN_PM_GAN_dropout_PD_y1_iter_{0}.pth"
            run_parameters["Model_DCN_PM_GAN_PD_y0"] = "./Models/PM_GAN_PD/DCN_PM_GAN_dropout_PD_y0_iter_{0}.pth"

            run_parameters["TARNET"] = "./MSE/ITE/ITE_TARNET_iter_{0}.csv"

            run_parameters["TARNET_PM_GAN"] = "./MSE/ITE/ITE_TARNET_PM_GAN_iter_{0}.csv"

            run_parameters["summary_file_name"] = "Details_original.txt"
            run_parameters["is_synthetic"] = False

        elif self.running_mode == "synthetic_data":
            run_parameters["input_nodes"] = 75
            run_parameters["consolidated_file_path"] = "./MSE_Augmented/Results_consolidated.csv"

            run_parameters["is_synthetic"] = True

        return run_parameters

    def __load_data(self, train_path, test_path, iter_id):
        if self.running_mode == "original_data":
            return self.dL.load_train_test_jobs(train_path, test_path, iter_id)

        elif self.running_mode == "synthetic_data":
            return self.dL.load_train_test_jobs(train_path, test_path, iter_id)

    def __get_ps_model(self, ps_model_type, iter_id,
                       input_nodes, device):
        ps_train_set = self.dL.convert_to_tensor(self.np_covariates_X_train, self.np_covariates_T_train)
        ps_test_set = self.dL.convert_to_tensor(self.np_covariates_X_test,
                                                self.np_covariates_T_test)
        ps_manager = PS_Manager()
        if ps_model_type == Constants.PS_MODEL_NN:
            return ps_manager.get_propensity_scores(ps_train_set,
                                                    ps_test_set, iter_id,
                                                    input_nodes, device)
        elif ps_model_type == Constants.PS_MODEL_LR:
            return ps_manager.get_propensity_scores_using_LR(self.np_covariates_X_train,
                                                             self.np_covariates_T_train,
                                                             self.np_covariates_X_test,
                                                             regularized=False)
        elif ps_model_type == Constants.PS_MODEL_LR_Lasso:
            return ps_manager.get_propensity_scores_using_LR(self.np_covariates_X_train,
                                                             self.np_covariates_T_train,
                                                             self.np_covariates_X_test,
                                                             regularized=True)

    @staticmethod
    def cal_policy_val(t, yf, eff_pred):
        #  policy_val(t[e>0], yf[e>0], eff_pred[e>0], compute_policy_curve)

        if np.any(np.isnan(eff_pred)):
            return np.nan, np.nan

        policy = eff_pred > 0
        treat_overlap = (policy == t) * (t > 0)
        control_overlap = (policy == t) * (t < 1)

        if np.sum(treat_overlap) == 0:
            treat_value = 0
        else:
            treat_value = np.mean(yf[treat_overlap])

        if np.sum(control_overlap) == 0:
            control_value = 0
        else:
            control_value = np.mean(yf[control_overlap])

        pit = np.mean(policy)
        policy_value = pit * treat_value + (1 - pit) * control_value

        return policy_value

    def __process_evaluated_metric(self, y_f, e, T,
                                   y1_hat, y0_hat,
                                   ite_dict, predicted_ITE_list,
                                   ite_csv_path,
                                   iter_id):
        y1_hat_np = np.array(y1_hat)
        y0_hat_np = np.array(y0_hat)
        e_np = np.array(e)
        t_np = np.array(T)
        np_y_f = np.array(y_f)

        y1_hat_np_b = 1.0 * (y1_hat_np > 0.5)
        y0_hat_np_b = 1.0 * (y0_hat_np > 0.5)

        err_fact = np.mean(np.abs(y1_hat_np_b - np_y_f))
        att = np.mean(np_y_f[t_np > 0]) - np.mean(np_y_f[(1 - t_np + e_np) > 1])

        eff_pred = y0_hat_np - y1_hat_np
        eff_pred[t_np > 0] = -eff_pred[t_np > 0]

        ate_pred = np.mean(eff_pred[e_np > 0])
        atc_pred = np.mean(eff_pred[(1 - t_np + e_np) > 1])

        att_pred = np.mean(eff_pred[(t_np + e_np) > 1])
        bias_att = np.abs(att_pred - att)

        policy_value = self.cal_policy_val(t_np[e_np > 0], np_y_f[e_np > 0],
                                           eff_pred[e_np > 0])

        print("bias_att: " + str(bias_att))
        print("policy_value: " + str(policy_value))
        print("Risk: " + str(1 - policy_value))
        print("atc_pred: " + str(atc_pred))
        print("att_pred: " + str(att_pred))
        print("err_fact: " + str(err_fact))

        Utils.write_to_csv(ite_csv_path.format(iter_id), ite_dict)
        return ate_pred, att_pred, bias_att, atc_pred, policy_value, 1 - policy_value, err_fact

    def get_consolidated_file_name(self, ps_model_type):
        if ps_model_type == Constants.PS_MODEL_NN:
            return "./MSE/Results_consolidated_NN.csv"
        elif ps_model_type == Constants.PS_MODEL_LR:
            return "./MSE/Results_consolidated_LR.csv"
        elif ps_model_type == Constants.PS_MODEL_LR_Lasso:
            return "./MSE/Results_consolidated_LR_LAsso.csv"