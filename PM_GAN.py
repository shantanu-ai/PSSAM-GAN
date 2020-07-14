from Constants import Constants
from DCN_network import DCN_network
from Propensity_socre_network import Propensity_socre_network
from Utils import Utils


class PM_GAN:
    def train_eval_DCN(self, iter_id, np_covariates_X_train,
                       np_covariates_Y_train,
                       dL, device,
                       run_parameters,
                       is_synthetic=False):
        print("----------- Training and evaluation phase ------------")
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

        # using NN
        MSE_NN, true_ATE_NN, predicted_ATE_NN = self.__test_DCN_NN(iter_id,
                                                                   np_covariates_X_test,
                                                                   np_covariates_Y_test,
                                                                   dL, device,
                                                                   ps_test_set,
                                                                   run_parameters["nn_prop_file"],
                                                                   run_parameters["nn_iter_file"],
                                                                   run_parameters["is_synthetic"],
                                                                   run_parameters["input_nodes"])

        return {
            "MSE_NN": MSE_NN,
            "true_ATE_NN": true_ATE_NN,
            "predicted_ATE_NN": predicted_ATE_NN,
        }

    def __train_propensity_net_NN(self, ps_train_set,
                                  np_covariates_X_train,
                                  np_covariates_Y_train, dL,
                                  iter_id, device, input_nodes, is_synthetic):
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

        # train DCN
        print("############### DCN Training using NN ###############")
        data_loader_dict_train_NN = dL.prepare_tensor_for_DCN(np_covariates_X_train,
                                                              np_covariates_Y_train,
                                                              ps_score_list_train_NN,
                                                              is_synthetic)

        model_path_PD = Constants.DCN_MODEL_PATH_PD
        self.__train_DCN(data_loader_dict_train_NN, iter_id, model_path_PD, dL, device,
                         input_nodes, train_mode=Constants.DCN_TRAIN_PD)

    def __train_DCN(self, data_loader_dict_train, iter_id, model_path, dL,
                    device, input_nodes, train_mode=Constants.DCN_TRAIN_PD):
        tensor_treated_train = Utils.create_tensors_to_train_DCN(data_loader_dict_train["treated_data"], dL)
        tensor_control_train = Utils.create_tensors_to_train_DCN(data_loader_dict_train["control_data"], dL)

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
        dcn.train(DCN_train_parameters, device, train_mode=Constants.DCN_TRAIN_PD)

    def __test_DCN_NN(self, iter_id,
                      np_covariates_X_test, np_covariates_Y_test,
                      dL, device, ps_test_set,
                      prop_score_file, iter_file, is_synthetic, input_nodes):
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

        # load data for ITE network using vanilla network
        print("############### DCN Testing using NN ###############")
        data_loader_dict_NN = dL.prepare_tensor_for_DCN(np_covariates_X_test,
                                                        np_covariates_Y_test,
                                                        ps_score_list_NN,
                                                        is_synthetic)
        model_path = Constants.DCN_MODEL_PATH_PD(iter_id,
                                                 Constants.DCN_EPOCHS,
                                                 Constants.DCN_LR)
        MSE_NN, true_ATE_NN, predicted_ATE_NN, ITE_dict_list = self.__do_test_DCN(data_loader_dict_NN,
                                                                                  dL, device,
                                                                                  model_path,
                                                                                  input_nodes)
        Utils.write_to_csv(iter_file.format(iter_id), ITE_dict_list)

        return MSE_NN, true_ATE_NN, predicted_ATE_NN

    @staticmethod
    def __do_test_DCN(data_loader_dict, dL, device, model_path, input_nodes):
        treated_group = data_loader_dict["treated_data"]
        np_treated_df_X = treated_group[0]
        np_treated_ps_score = treated_group[1]
        np_treated_df_Y_f = treated_group[2]
        np_treated_df_Y_cf = treated_group[3]
        tensor_treated = dL.convert_to_tensor_DCN(np_treated_df_X, np_treated_ps_score,
                                                  np_treated_df_Y_f, np_treated_df_Y_cf)

        control_group = data_loader_dict["control_data"]
        np_control_df_X = control_group[0]
        np_control_ps_score = control_group[1]
        np_control_df_Y_f = control_group[2]
        np_control_df_Y_cf = control_group[3]
        tensor_control = dL.convert_to_tensor_DCN(np_control_df_X, np_control_ps_score,
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
