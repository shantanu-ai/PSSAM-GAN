from Constants import Constants
from DCN_network import DCN_network
from PS_Manager import PS_Manager
from Propensity_socre_network import Propensity_socre_network
from Utils import Utils


class PM_GAN:
    def train_eval_DCN(self, iter_id, np_covariates_X_train,
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

        # testing using NN
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
        MSE_NN_PD = 0
        true_ATE_NN_PD = 0
        predicted_ATE_NN_PD = 0

        MSE_NN_dropout_2 = 0
        true_ATE_NN_dropout_2 = 0
        predicted_ATE_NN_dropout_2 = 0

        MSE_NN_dropout_5 = 0
        true_ATE_NN_dropout_5 = 0
        predicted_ATE_NN_dropout_5 = 0

        # test using PD - DCN-PD - Model 1
        # model_path = Constants.DCN_MODEL_PATH_PD.format(iter_id,
        #                                                 Constants.DCN_EPOCHS,
        #                                                 Constants.DCN_LR)
        # print("############### DCN Testing using NN PD ###############")
        # print(model_path)
        # print("--" * 25)
        # MSE_NN_PD, true_ATE_NN_PD, predicted_ATE_NN_PD = self.__test_DCN_NN(iter_id,
        #                                                                     dL,
        #                                                                     device,
        #                                                                     data_loader_dict_NN,
        #                                                                     run_parameters["nn_DCN_PD_iter_file"],
        #                                                                     model_path,
        #                                                                     input_nodes)

        # test using no dropout - PM Match - Model 2
        model_path = Constants.DCN_MODEL_PATH_NO_DROPOUT_PM_MATCH_FALSE.format(iter_id,
                                                                               Constants.DCN_EPOCHS,
                                                                               Constants.DCN_LR)
        print("############### DCN Testing using NN no Dropout ###############")
        print(model_path)
        print("--" * 25)
        MSE_NN_no_dropout, true_ATE_NN_no_dropout, predicted_ATE_NN_no_dropout = \
            self.__test_DCN_NN(iter_id,
                               dL,
                               device,
                               data_loader_dict_NN,
                               run_parameters[
                                   "nn_DCN_No_Dropout_iter_file"],
                               model_path,
                               input_nodes)
        model_path = Constants.DCN_MODEL_PATH_NO_DROPOUT_PM_MATCH_TRUE.format(iter_id,
                                                                               Constants.DCN_EPOCHS,
                                                                               Constants.DCN_LR)
        MSE_NN_no_dropout_pm_match, true_ATE_NN_no_dropout_pm_match, predicted_ATE_NN_no_dropout_pm_match = \
            self.__test_DCN_NN(iter_id,
                               dL,
                               device,
                               data_loader_dict_NN,
                               run_parameters[
                                   "nn_DCN_No_Dropout_iter_file"],
                               model_path,
                               input_nodes)

        # # using NN Constant Dropout 0.5 Model 3
        # model_path = Constants.DCN_MODEL_PATH_CONSTANT_DROPOUT_5.format(iter_id,
        #                                                                 Constants.DCN_EPOCHS,
        #                                                                 Constants.DCN_LR)
        # print("############### DCN Testing using NN Constant dropout 0.5 ###############")
        # print(model_path)
        # print("--" * 25)
        # MSE_NN_dropout_5, true_ATE_NN_dropout_5, predicted_ATE_NN_dropout_5 = \
        #     self.__test_DCN_NN(iter_id,
        #                        dL,
        #                        device,
        #                        data_loader_dict_NN,
        #                        run_parameters[
        #                            "nn_DCN_Const_Dropout_5_iter_file"],
        #                        model_path,
        #                        input_nodes)
        #
        # # using NN Constant Dropout 0.2 Model 4
        # model_path = Constants.DCN_MODEL_PATH_CONSTANT_DROPOUT_2.format(iter_id,
        #                                                                 Constants.DCN_EPOCHS,
        #                                                                 Constants.DCN_LR)
        # print("############### DCN Testing using NN NN Constant dropout 0.2 ###############")
        # print(model_path)
        # print("--" * 25)
        # MSE_NN_dropout_2, true_ATE_NN_dropout_2, predicted_ATE_NN_dropout_2 = \
        #     self.__test_DCN_NN(iter_id,
        #                        dL,
        #                        device,
        #                        data_loader_dict_NN,
        #                        run_parameters[
        #                            "nn_DCN_Const_Dropout_2_iter_file"],
        #                        model_path,
        #                        input_nodes)

        return {
            "MSE_NN_PD": MSE_NN_PD,
            "true_ATE_NN_PD": true_ATE_NN_PD,
            "predicted_ATE_NN_PD": predicted_ATE_NN_PD,

            "MSE_NN_no_dropout": MSE_NN_no_dropout,
            "true_ATE_NN_no_dropout": true_ATE_NN_no_dropout,
            "predicted_ATE_NN_no_dropout": predicted_ATE_NN_no_dropout,

            "MSE_NN_dropout_5": MSE_NN_dropout_5,
            "true_ATE_NN_dropout_5": true_ATE_NN_dropout_5,
            "predicted_ATE_NN_dropout_5": predicted_ATE_NN_dropout_5,

            "MSE_NN_dropout_2": MSE_NN_dropout_2,
            "true_ATE_NN_dropout_2": true_ATE_NN_dropout_2,
            "predicted_ATE_NN_dropout_2": predicted_ATE_NN_dropout_2
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

        # train using PD - DCN-PD Model 1
        # need to change the model path name to incorporate ps_match=True
        # model_path_PD = Constants.DCN_MODEL_PATH_PD
        # print("############### DCN Training using NN PD (No PS Match) ###############")
        # print("--" * 25)
        # self.__train_DCN(data_loader_dict_train_NN, iter_id, model_path_PD, dL, device,
        #                  input_nodes, train_mode=Constants.DCN_TRAIN_PD,
        #                  ps_match=True)
        #
        # model_path_PD = Constants.DCN_MODEL_PATH_PD
        # print("############### DCN Training using NN PD (No PS Match) ###############")
        # print("--" * 25)
        # self.__train_DCN(data_loader_dict_train_NN, iter_id, model_path_PD, dL, device,
        #                  input_nodes, train_mode=Constants.DCN_TRAIN_PD,
        #                  ps_match=False)

        # train using no dropout Model 2
        model_path_no_dropout = Constants.DCN_MODEL_PATH_NO_DROPOUT_PM_MATCH_TRUE
        print("############### DCN Training using NN No Dropout (PS Match) ###############")
        print("--" * 25)
        self.__train_DCN(data_loader_dict_train_NN, iter_id,
                         model_path_no_dropout,
                         dL, device,
                         input_nodes,
                         train_mode=Constants.DCN_TRAIN_NO_DROPOUT,
                         ps_match=True)

        model_path_no_dropout = Constants.DCN_MODEL_PATH_NO_DROPOUT_PM_MATCH_FALSE
        print("############### DCN Training using NN No Dropout (PS Match) ###############")
        print("--" * 25)
        self.__train_DCN(data_loader_dict_train_NN, iter_id,
                         model_path_no_dropout,
                         dL, device,
                         input_nodes,
                         train_mode=Constants.DCN_TRAIN_NO_DROPOUT,
                         ps_match=False)

        # train using constant dropout 0.5 Model 3
        # model_path_constant_dropout = Constants.DCN_MODEL_PATH_CONSTANT_DROPOUT_5
        # print("############### DCN Training using NN Dropout 0.5 (PS Match) ###############")
        # print("--" * 25)
        # self.__train_DCN(data_loader_dict_train_NN, iter_id,
        #                  model_path_constant_dropout,
        #                  dL, device,
        #                  input_nodes,
        #                  train_mode=Constants.DCN_TRAIN_CONSTANT_DROPOUT_5)
        #
        # # train using constant dropout 0.2 Model 4
        # model_path_constant_dropout = Constants.DCN_MODEL_PATH_CONSTANT_DROPOUT_2
        # print("############### DCN Training using NN Dropout 0.2 (PS Match) ###############")
        # print("--" * 25)
        # self.__train_DCN(data_loader_dict_train_NN, iter_id,
        #                  model_path_constant_dropout, dL, device,
        #                  input_nodes,
        #                  train_mode=Constants.DCN_TRAIN_CONSTANT_DROPOUT_2)

    def __train_DCN(self, data_loader_dict_train, iter_id, model_path, dL,
                    device, input_nodes, train_mode=Constants.DCN_TRAIN_PD, ps_match=True):

        if ps_match:
            psm = PS_Manager()
            tensor_treated_train, tensor_control_train = \
                psm.match_using_prop_score(data_loader_dict_train["treated_data"],
                                           data_loader_dict_train["control_data"], dL)

        else:
            tensor_treated_train = \
                Utils.create_tensors_to_train_DCN(data_loader_dict_train["treated_data"], dL)
            tensor_control_train = \
                Utils.create_tensors_to_train_DCN(data_loader_dict_train["control_data"], dL)

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

    def __test_DCN_NN(self, iter_id,
                      dL,
                      device,
                      data_loader_dict_NN,
                      iter_file,
                      model_path,
                      input_nodes):
        MSE_NN_PD, true_ATE_NN_PD, predicted_ATE_NN_PD, ITE_dict_list = self.__do_test_DCN(data_loader_dict_NN,
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
