import numpy as np

from Constants import Constants
from DCN_network import DCN_network
from PS_Matching import PS_Matching
from Propensity_socre_network import Propensity_socre_network
from Utils import Utils


class PM_GAN:
    def train_DCN(self, iter_id, np_covariates_X_train,
                  np_covariates_Y_train,
                  dL, device,
                  run_parameters,
                  is_synthetic=False):
        print("----------- Training phase ------------")
        ps_train_set = dL.convert_to_tensor(np_covariates_X_train, np_covariates_Y_train)

        # using NN
        self.__train_propensity_net_NN(ps_train_set,
                                       np_covariates_X_train,
                                       np_covariates_Y_train,
                                       dL,
                                       iter_id, device,
                                       run_parameters["input_nodes"],
                                       is_synthetic)

    def test_DCN(self, iter_id, np_covariates_X_test, np_covariates_Y_test,
                 dL,
                 device,
                 run_parameters):
        print("----------- Testing phase ------------")
        ps_test_set = dL.convert_to_tensor(np_covariates_X_test,
                                           np_covariates_Y_test)

        prop_score_file = run_parameters["nn_prop_file"]
        is_synthetic = run_parameters["is_synthetic"]
        input_nodes = run_parameters["input_nodes"]

        # get propensity scores using NN
        ps_net_NN = Propensity_socre_network()
        ps_eval_parameters_NN = {
            "eval_set": ps_test_set,
            "model_path": Constants.PROP_SCORE_NN_MODEL_PATH
                .format(iter_id, Constants.PROP_SCORE_NN_EPOCHS, Constants.PROP_SCORE_NN_LR),
            "input_nodes": input_nodes
        }
        ps_score_list_NN = ps_net_NN.eval(ps_eval_parameters_NN, device, phase="eval")
        Utils.write_to_csv(prop_score_file.format(iter_id), ps_score_list_NN)

        data_loader_dict_NN = dL.prepare_tensor_for_DCN(np_covariates_X_test,
                                                        np_covariates_Y_test,
                                                        ps_score_list_NN,
                                                        is_synthetic)

        # test using PM Match No PD - Model 1
        # model_path = Constants.DCN_MODEL_PATH_NO_DROPOUT_PM_MATCH_TRUE \
        #     .format(iter_id,
        #             Constants.DCN_EPOCHS,
        #             Constants.DCN_LR)
        # print("############### DCN Testing using PM Match No PD ###############")
        # print("--" * 25)
        # print(model_path)
        # print("--" * 25)
        # MSE_PM_Match_No_PD, true_ATE_PM_Match_No_PD, predicted_ATE_PM_Match_No_PD = \
        #     self.__test_DCN_NN(iter_id,
        #                        dL,
        #                        device,
        #                        data_loader_dict_NN,
        #                        run_parameters[
        #                            "nn_DCN_PS_Match_No_PD_iter_file"],
        #                        model_path,
        #                        input_nodes)

        # test using PM Match PD - Model 2
        # model_path = Constants.DCN_MODEL_PATH_PD_PM_MATCH_TRUE.format(iter_id,
        #                                                               Constants.DCN_EPOCHS,
        #                                                               Constants.DCN_LR)
        # print("--" * 25)
        # print("############### Model 1: DCN Testing using PM Match PD  ###############")
        # print(model_path)
        # print("--" * 25)
        # MSE_PM_Match_PD, true_ATE_PM_Match_PD, predicted_ATE_PM_Match_PD = \
        #     self.__test_DCN_NN(iter_id,
        #                        dL,
        #                        device,
        #                        data_loader_dict_NN,
        #                        run_parameters["nn_DCN_PS_Match_PD_iter_file"],
        #                        model_path,
        #                        input_nodes)

        # test using No PM Match PD - Model 3
        model_path = Constants.DCN_MODEL_PATH_PD_PM_MATCH_FALSE.format(iter_id,
                                                                       Constants.DCN_EPOCHS,
                                                                       Constants.DCN_LR)
        print("--" * 25)
        print("############### Model 2: DCN Testing using No PM Match PD (DCN-PD)  ###############")
        print(model_path)
        print("--" * 25)
        MSE_No_PM_Match_PD, true_ATE_No_PM_Match_PD, predicted_ATE_No_PM_Match_PD = \
            self.__test_DCN_NN(iter_id,
                               dL,
                               device,
                               data_loader_dict_NN,
                               run_parameters["nn_DCN_No_PS_Match_PD_iter_file"],
                               model_path,
                               input_nodes)

        return {
            "MSE_PM_Match_No_PD": 0,
            "true_ATE_PM_Match_No_PD": 0,
            "predicted_ATE_PM_Match_No_PD": 0,

            "MSE_PM_Match_PD": 0,
            "true_ATE_PM_Match_PD": 0,
            "predicted_ATE_PM_Match_PD": 0,

            "MSE_No_PM_Match_PD": MSE_No_PM_Match_PD,
            "true_ATE_No_PM_Match_PD": true_ATE_No_PM_Match_PD,
            "predicted_ATE_No_PM_Match_PD": predicted_ATE_No_PM_Match_PD
        }

    def __train_propensity_net_NN(self, ps_train_set,
                                  np_covariates_X_train,
                                  np_covariates_Y_train, dL,
                                  iter_id, device, input_nodes, is_synthetic):
        # get propensity scores
        ps_score_list_train_NN = self.get_propensity_scores(ps_train_set, iter_id, input_nodes, device)

        data_loader_dict_train_NN = dL.prepare_tensor_for_DCN(np_covariates_X_train,
                                                              np_covariates_Y_train,
                                                              ps_score_list_train_NN,
                                                              is_synthetic)

        # train using PM Match No PD Model 1
        # model_path_no_dropout = Constants.DCN_MODEL_PATH_NO_DROPOUT_PM_MATCH_TRUE
        # print("--" * 25)
        # print("############### DCN Training using PM Match No PD ###############")
        # print("--" * 25)
        # self.__train_DCN(data_loader_dict_train_NN, iter_id,
        #                  model_path_no_dropout,
        #                  dL, device,
        #                  input_nodes,
        #                  train_mode=Constants.DCN_TRAIN_NO_DROPOUT,
        #                  ps_match=True)

        # train using PM Match PD Model 2
        # model_path_PD = Constants.DCN_MODEL_PATH_PD_PM_MATCH_TRUE
        # print("--" * 25)
        # print("############### Model 1: DCN Training using PM Match PD ###############")
        # print("--" * 25)
        # self.__train_DCN(data_loader_dict_train_NN, iter_id,
        #                  model_path_PD,
        #                  dL, device,
        #                  input_nodes,
        #                  train_mode=Constants.DCN_TRAIN_PD,
        #                  ps_match=True)

        # train using No PM Match PD (DCN-PD) Model 3
        model_path_PD = Constants.DCN_MODEL_PATH_PD_PM_MATCH_FALSE
        print("--" * 25)
        print("############### Model 2: DCN Training using No PM Match PD (DCN-PD) ###############")
        print("--" * 25)
        self.__train_DCN(data_loader_dict_train_NN, iter_id,
                         model_path_PD,
                         dL,
                         device,
                         input_nodes,
                         train_mode=Constants.DCN_TRAIN_PD,
                         ps_match=False)

    def __train_DCN(self, data_loader_dict_train, iter_id, model_path, dL,
                    device, input_nodes, train_mode=Constants.DCN_TRAIN_PD, ps_match=True):
        if ps_match:
            tensor_treated_train, tensor_control_train = \
                self.__execute_PM_training(iter_id,
                                           data_loader_dict_train,
                                           dL,
                                           input_nodes,
                                           train_mode,
                                           device)

        else:
            tensor_treated_train = \
                Utils.create_tensors_to_train_DCN(data_loader_dict_train["treated_data"], dL)
            tensor_control_train = \
                Utils.create_tensors_to_train_DCN(data_loader_dict_train["control_data"], dL)

        self.__execute_DCN_train(tensor_treated_train, tensor_control_train, model_path, iter_id,
                                 input_nodes, device, train_mode)

    @staticmethod
    def __execute_DCN_train(tensor_treated_train, tensor_control_train, model_path, iter_id,
                            input_nodes, device, train_mode):
        DCN_train_parameters = {
            "epochs": Constants.DCN_EPOCHS,
            "lr": Constants.DCN_LR,
            "treated_batch_size": 1,
            "control_batch_size": 1,
            "shuffle": True,
            "treated_set_train": tensor_treated_train,
            "control_set_train": tensor_control_train,
            "model_save_path": model_path.format(iter_id,
                                                 Constants.DCN_EPOCHS,
                                                 Constants.DCN_LR),
            "input_nodes": input_nodes
        }

        # train DCN network
        dcn = DCN_network()
        dcn.train(DCN_train_parameters, device, train_mode=train_mode)

    def __execute_PM_training(self, iter_id, data_loader_dict_train, dL, input_nodes,
                              train_mode, device):
        tuple_treated = data_loader_dict_train["treated_data"]
        tuple_control = data_loader_dict_train["control_data"]

        psm = PS_Matching()
        prop_score_NN_model_path = Constants.PROP_SCORE_NN_MODEL_PATH \
            .format(iter_id, Constants.PROP_SCORE_NN_EPOCHS, Constants.PROP_SCORE_NN_LR)

        gan_response = \
            psm.match_using_prop_score(tuple_treated,
                                       tuple_control, dL,
                                       prop_score_NN_model_path,
                                       device)
        tuple_unmatched_control = gan_response["tuple_unmatched_control"]
        tuple_matched_control = gan_response["tuple_matched_control"]
        treated_generated = gan_response["treated_generated"]
        ps_score_list_treated = gan_response["ps_score_list_treated"]

        print("### DCN semi supervised training using PS Matching ###")
        tensor_treated = \
            Utils.create_tensors_to_train_DCN(tuple_treated, dL)
        tensor_matched_control = \
            Utils.create_tensors_to_train_DCN(tuple_matched_control, dL)

        tensor_all_control = Utils.create_tensors_to_train_DCN(
            data_loader_dict_train["control_data"], dL)

        model_path_semi_supervised = Constants.DCN_MODEL_SEMI_SUPERVISED_PATH
        self.__execute_DCN_train(tensor_treated, tensor_all_control, model_path_semi_supervised,
                                 iter_id,
                                 input_nodes, device, train_mode,
                                 epochs=100)

        print("### DCN supervised evaluation for GAN generated treated samples ###")
        ps_score_list_treated_np = np.array(ps_score_list_treated)
        eval_set = Utils.convert_to_tensor_DCN_PS(treated_generated.detach().cpu(),
                                                  ps_score_list_treated_np)
        DCN_test_parameters = {
            "eval_set": eval_set,
            "model_save_path": model_path_semi_supervised.format(iter_id,
                                                                 Constants.DCN_EPOCHS,
                                                                 Constants.DCN_LR)
        }
        dcn = DCN_network()
        response_dict = dcn.eval_semi_supervised(DCN_test_parameters, device, input_nodes,
                                                 Constants.DCN_EVALUATION, treated_flag=True)

        tensor_treated_all = self.get_treated_tensor_all_ds(treated_generated, ps_score_list_treated_np,
                                                            response_dict,
                                                            tuple_treated, device)

        tensor_control_all = self.get_control_all_ds(tuple_unmatched_control, tuple_matched_control)

        print("### DCN training using all dataset ###")
        # return tensor_treated_all, tensor_control_all

        return tensor_treated_all, tensor_all_control

    @staticmethod
    def get_treated_tensor_all_ds(treated_generated, ps_score_list_treated_np, response_dict,
                                  tuple_treated, device):
        np_treated_generated = treated_generated.detach().cpu().numpy()
        np_ps_score_list_gen_treated = ps_score_list_treated_np
        np_treated_gen_f = Utils.convert_to_col_vector(response_dict["y_f_list"])
        np_treated_gen_cf = Utils.convert_to_col_vector(response_dict["y_cf_list"])

        np_original_X = tuple_treated[0]
        np_original_ps_score = tuple_treated[1]
        np_original_Y_f = tuple_treated[2]
        np_original_Y_cf = tuple_treated[3]

        np_treated_x = np.concatenate((np_treated_generated, np_original_X), axis=0)
        np_treated_ps = np.concatenate((np_ps_score_list_gen_treated, np_original_ps_score), axis=0)
        np_treated_yf = np.concatenate((np_treated_gen_f, np_original_Y_f), axis=0)
        np_treated_y_cf = np.concatenate((np_treated_gen_cf, np_original_Y_cf), axis=0)

        tensor_treated = Utils.convert_to_tensor_DCN(np_treated_x, np_treated_ps,
                                                     np_treated_yf, np_treated_y_cf)

        return tensor_treated

    @staticmethod
    def get_control_all_ds(tuple_unmatched_control, tuple_matched_control):
        np_control_unmatched_X = tuple_unmatched_control[0]
        np_ps_score_list_control_unmatched = tuple_unmatched_control[1]
        np_control_unmatched_f = tuple_unmatched_control[2]
        np_control_unmatched_cf = tuple_unmatched_control[3]

        np_control_matched_X = tuple_matched_control[0]
        np_ps_score_list_control_matched = tuple_matched_control[1]
        np_control_matched_f = tuple_matched_control[2]
        np_control_matched_cf = tuple_matched_control[3]

        np_control_x = np.concatenate((np_control_unmatched_X, np_control_matched_X), axis=0)
        np_control_ps = np.concatenate((np_ps_score_list_control_unmatched, np_ps_score_list_control_matched), axis=0)
        np_control_f = np.concatenate((np_control_unmatched_f, np_control_matched_f), axis=0)
        np_control_cf = np.concatenate((np_control_unmatched_cf, np_control_matched_cf), axis=0)

        tensor_control = Utils.convert_to_tensor_DCN(np_control_x, np_control_ps,
                                                     np_control_f, np_control_cf)

        return tensor_control

    def __test_DCN_NN(self, iter_id,
                      dL,
                      device,
                      data_loader_dict_NN,
                      iter_file,
                      model_path,
                      input_nodes):
        MSE_NN_PD, true_ATE_NN_PD, predicted_ATE_NN_PD, ITE_dict_list = \
            self.__do_test_DCN(data_loader_dict_NN,
                               dL, device,
                               model_path,
                               input_nodes)
        Utils.write_to_csv(iter_file.format(iter_id), ITE_dict_list)

        return MSE_NN_PD, true_ATE_NN_PD, predicted_ATE_NN_PD

    @staticmethod
    def __do_test_DCN(data_loader_dict, dL, device, model_path, input_nodes):
        treated_group = data_loader_dict["treated_data"]
        np_treated_df_X = treated_group[0]
        np_treated_ps_score = treated_group[1]
        np_treated_df_Y_f = treated_group[2]
        np_treated_df_Y_cf = treated_group[3]
        tensor_treated = Utils.convert_to_tensor_DCN(np_treated_df_X, np_treated_ps_score,
                                                     np_treated_df_Y_f, np_treated_df_Y_cf)

        control_group = data_loader_dict["control_data"]
        np_control_df_X = control_group[0]
        np_control_ps_score = control_group[1]
        np_control_df_Y_f = control_group[2]
        np_control_df_Y_cf = control_group[3]
        tensor_control = Utils.convert_to_tensor_DCN(np_control_df_X, np_control_ps_score,
                                                     np_control_df_Y_f, np_control_df_Y_cf)

        DCN_test_parameters = {
            "treated_set": tensor_treated,
            "control_set": tensor_control,
            "model_save_path": model_path
        }

        dcn = DCN_network()
        response_dict = dcn.eval(DCN_test_parameters, device, input_nodes,
                                 Constants.DCN_EVALUATION)
        err_treated = [ele ** 2 for ele in response_dict["treated_err"]]
        err_control = [ele ** 2 for ele in response_dict["control_err"]]

        true_ATE = sum(response_dict["true_ITE"]) / len(response_dict["true_ITE"])
        predicted_ATE = sum(response_dict["predicted_ITE"]) / len(response_dict["predicted_ITE"])

        total_sum = sum(err_treated) + sum(err_control)
        total_item = len(err_treated) + len(err_control)
        MSE = total_sum / total_item

        max_treated = max(err_treated)
        max_control = max(err_control)
        max_total = max(max_treated, max_control)

        min_treated = min(err_treated)
        min_control = min(err_control)
        min_total = min(min_treated, min_control)

        print("MSE: {0}".format(MSE))
        print("Max: {0}, Min: {1}".format(max_total, min_total))

        return MSE, true_ATE, predicted_ATE, response_dict["ITE_dict_list"]

    def get_propensity_scores(self, ps_train_set, iter_id, input_nodes, device):
        prop_score_NN_model_path = Constants.PROP_SCORE_NN_MODEL_PATH \
            .format(iter_id, Constants.PROP_SCORE_NN_EPOCHS, Constants.PROP_SCORE_NN_LR)

        train_parameters_NN = {
            "epochs": Constants.PROP_SCORE_NN_EPOCHS,
            "lr": Constants.PROP_SCORE_NN_LR,
            "batch_size": Constants.PROP_SCORE_NN_BATCH_SIZE,
            "shuffle": True,
            "train_set": ps_train_set,
            "model_save_path": prop_score_NN_model_path,
            "input_nodes": input_nodes
        }

        # ps using NN
        ps_net_NN = Propensity_socre_network()
        print("############### Propensity Score neural net Training ###############")
        ps_net_NN.train(train_parameters_NN, device, phase="train")

        # eval
        eval_parameters_train_NN = {
            "eval_set": ps_train_set,
            "model_path": prop_score_NN_model_path,
            "input_nodes": input_nodes
        }

        ps_score_list_train_NN = ps_net_NN.eval(eval_parameters_train_NN, device, phase="eval")

        return ps_score_list_train_NN
