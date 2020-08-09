import numpy as np

from Constants import Constants
from TARNet_Manager import TARNet_Manager
from Utils import Utils


class TARNet_Experiments:
    def __init__(self, input_nodes, device):
        self.data_loader_dict_test = None
        self.input_nodes = input_nodes
        self.device = device

    def evaluate_TARNet_Model(self, tuple_treated_train_original, tuple_control_train_original,
                              tensor_treated_balanced, tuple_control_balanced_tarnet,
                              data_loader_dict_test):
        # data loader -> (np_treated_df_X, np_treated_ps_score, np_treated_df_Y_f, np_treated_df_Y_cf)

        self.data_loader_dict_test = data_loader_dict_test

        # Model 1: TARNET
        print("--" * 20)
        print("###### Model 1: TARNET Supervised Training started ######")
        tarnet_pd_eval_dict = self.evaluate_TARNet(tuple_treated_train_original,
                                                   tuple_control_train_original)

        # Model 2: TARNET PM GAN
        print("--" * 20)
        print("###### Model 2: TARNET PM GAN Supervised Training started ######")
        tarnet_pm_gan_eval_dict = self.evaluate_TARNet_PM_GAN(tensor_treated_balanced,
                                                              tuple_control_balanced_tarnet)

        return {
            "tarnet_pd_eval_dict": tarnet_pd_eval_dict,
            "tarnet_pm_gan_eval_dict": tarnet_pm_gan_eval_dict
        }

    def evaluate_TARNet(self, tuple_treated_train_original,
                        tuple_control_train_original):
        np_treated_x, np_treated_ps, np_treated_f, np_treated_cf = tuple_treated_train_original
        np_control_x, np_control_ps, np_control_f, np_control_cf = tuple_control_train_original
        t_1 = np.ones(np_treated_x.shape[0])
        t_0 = np.zeros(np_control_x.shape[0])

        n_treated = np_treated_x.shape[0]
        n_control = np_control_x.shape[0]
        n_total = n_treated + n_control

        np_train_ss_X = np.concatenate((np_treated_x, np_control_x), axis=0)
        np_train_ss_ps = np.concatenate((np_treated_ps, np_control_ps), axis=0)
        np_train_ss_T = np.concatenate((t_1, t_0), axis=0)
        np_train_ss_f = np.concatenate((np_treated_f, np_control_f), axis=0)
        np_train_ss_cf = np.concatenate((np_treated_cf, np_control_cf), axis=0)

        train_set = Utils.create_tensors_to_train_DCN_semi_supervised((np_train_ss_X, np_train_ss_ps,
                                                                       np_train_ss_T, np_train_ss_f,
                                                                       np_train_ss_cf))
        train_parameters = self.__get_train_parameters(train_set)
        tarnet = TARNet_Manager(input_nodes=Constants.TARNET_INPUT_NODES,
                                shared_nodes=Constants.TARNET_SHARED_NODES,
                                outcome_nodes=Constants.TARNET_OUTPUT_NODES,
                                device=self.device)

        tarnet.train(train_parameters, self.device)

        tensor_treated_test = \
            Utils.create_tensors_from_tuple(self.data_loader_dict_test["treated_data"])
        tensor_control_test = \
            Utils.create_tensors_from_tuple(self.data_loader_dict_test["control_data"])
        _test_parameters = self.__get_test_parameters(tensor_treated_test, tensor_control_test)

        tarnet_eval_dict = tarnet.eval(_test_parameters, self.device)

        return tarnet_eval_dict

    def semi_supervised_train_eval(self, train_set,
                                   eval_set, n_total, n_treated):
        _train_parameters = self.__get_train_parameters(train_set)
        tarnet_ss = TARNet_Manager(input_nodes=Constants.TARNET_INPUT_NODES,
                                   shared_nodes=Constants.TARNET_SHARED_NODES,
                                   outcome_nodes=Constants.TARNET_OUTPUT_NODES,
                                   device=self.device)
        tarnet_ss.train_semi_supervised(_train_parameters, n_total, n_treated,
                                        self.device)

        _test_parameters = {
            "tensor_dataset": eval_set
        }
        return tarnet_ss.eval_semi_supervised(_test_parameters, self.device,
                                              treated_flag=True)

    def evaluate_TARNet_PM_GAN(self, tensor_treated_train, tuple_control_balanced_tarnet):
        _train_parameters = self.__get_train_parameters_supervised(tensor_treated_train,
                                                                   tuple_control_balanced_tarnet)

        tensor_treated_test = \
            Utils.create_tensors_from_tuple(self.data_loader_dict_test["treated_data"])
        tensor_control_test = \
            Utils.create_tensors_from_tuple(self.data_loader_dict_test["control_data"])
        _test_parameters = self.__get_test_parameters(tensor_treated_test, tensor_control_test)
        return self.__supervised_train_eval(_train_parameters,
                                            _test_parameters)

    def __supervised_train_eval(self, train_mode, _train_parameters,
                                _test_parameters):
        tarnet = TARNet_Manager(input_nodes=Constants.TARNET_INPUT_NODES,
                                shared_nodes=Constants.TARNET_SHARED_NODES,
                                outcome_nodes=Constants.TARNET_OUTPUT_NODES,
                                device=self.device)

        tarnet.train(_train_parameters, self.device)
        tarnet_eval_dict = tarnet.eval(_test_parameters, self.device)
        return tarnet_eval_dict

    @staticmethod
    def __get_train_parameters(train_set):
        return {
            "epochs": Constants.TARNET_EPOCHS,
            "lr": Constants.TARNET_LR,
            "lambda": Constants.TARNET_LAMBDA,
            "batch_size": Constants.TARNET_BATCH_SIZE,
            "shuffle": True,
            "tensor_dataset": train_set
        }

    @staticmethod
    def __get_train_parameters_supervised(tensor_treated, tuple_control_train):
        return {
            "epochs": Constants.TARNET_EPOCHS,
            "lr": Constants.TARNET_LR,
            "lambda": Constants.TARNET_LAMBDA,
            "batch_size": Constants.TARNET_BATCH_SIZE,
            "treated_tensor_dataset": tensor_treated,
            "tuple_control_train": tuple_control_train
        }

    @staticmethod
    def __get_test_parameters(tensor_treated_test, tensor_control_test):
        return {
            "treated_set": tensor_treated_test,
            "control_set": tensor_control_test
        }
