import numpy as np

from Constants import Constants
from TARNet_Manager import TARNet_Manager
from Utils import Utils


class TARNet_Experiments:
    def __init__(self, input_nodes, device):
        self.data_loader_dict_test = None
        self.data_loader_dict_val = None
        self.input_nodes = input_nodes
        self.device = device

    def semi_supervised_train_eval(self, train_set,
                                   data_loader_dict_val,
                                   eval_set, n_total, n_treated):
        _train_parameters = self.__get_train_parameters(train_set)
        tensor_treated_val = \
            Utils.create_tensors_from_tuple(data_loader_dict_val["treated_data"])
        tensor_control_val = \
            Utils.create_tensors_from_tuple(data_loader_dict_val["control_data"])
        val_parameters = self.__get_test_parameters(tensor_treated_val, tensor_control_val)

        tarnet_ss = TARNet_Manager(input_nodes=Constants.TARNET_INPUT_NODES,
                                   shared_nodes=Constants.TARNET_SHARED_NODES,
                                   outcome_nodes=Constants.TARNET_OUTPUT_NODES,
                                   device=self.device)
        tarnet_ss.train_semi_supervised(_train_parameters, val_parameters, n_total, n_treated,
                                        self.device)

        _test_parameters = {
            "tensor_dataset": eval_set
        }
        return tarnet_ss.eval_semi_supervised(_test_parameters, self.device,
                                              treated_flag=True)

    def evaluate_TARNet_Model(self, tuple_treated_train_original,
                              tuple_control_train_original,
                              tensor_treated_balanced,
                              data_loader_dict_val,
                              data_loader_dict_test,
                              n_total_balanced_tarnet,
                              n_treated_balanced_tarnet):
        # data loader -> (np_treated_df_X, np_treated_ps_score, np_treated_df_Y_f, np_treated_df_Y_cf)

        self.data_loader_dict_test = data_loader_dict_test
        self.data_loader_dict_val = data_loader_dict_val

        # Model 1: TARNET
        print("--" * 20)
        print("###### Model 1: TARNET Supervised Training started ######")
        print("Treated: "+str(tuple_treated_train_original[0].shape[0]))
        print("Control: "+str(tuple_control_train_original[0].shape[0]))
        tarnet_eval_dict = self.evaluate_TARNet(tuple_treated_train_original,
                                                tuple_control_train_original)

        # Model 2: TARNET PM GAN
        print("--" * 20)
        print("###### Model 2: TARNET PM GAN Supervised Training started ######")
        print("Treated: "+str(n_treated_balanced_tarnet))
        print("Control: "+str(n_total_balanced_tarnet - n_treated_balanced_tarnet))
        tarnet_pm_gan_eval_dict = self.evaluate_TARNet_PM_GAN(tensor_treated_balanced,
                                                              n_total_balanced_tarnet,
                                                              n_treated_balanced_tarnet)
        return {
            "tarnet_eval_dict": tarnet_eval_dict,
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

        tensor_treated_val = \
            Utils.create_tensors_from_tuple(self.data_loader_dict_val["treated_data"])
        tensor_control_val = \
            Utils.create_tensors_from_tuple(self.data_loader_dict_val["control_data"])
        val_parameters = self.__get_test_parameters(tensor_treated_val, tensor_control_val)

        tarnet = TARNet_Manager(input_nodes=Constants.TARNET_INPUT_NODES,
                                shared_nodes=Constants.TARNET_SHARED_NODES,
                                outcome_nodes=Constants.TARNET_OUTPUT_NODES,
                                device=self.device)

        tarnet.train_semi_supervised(train_parameters, val_parameters, n_total, n_treated, self.device)

        tensor_treated_test = \
            Utils.create_tensors_from_tuple(self.data_loader_dict_test["treated_data"])
        tensor_control_test = \
            Utils.create_tensors_from_tuple(self.data_loader_dict_test["control_data"])
        _test_parameters = self.__get_test_parameters(tensor_treated_test, tensor_control_test)

        tarnet_eval_dict = tarnet.eval(_test_parameters, self.device)

        return tarnet_eval_dict

    def evaluate_TARNet_PM_GAN(self, tensor_treated_train,
                               n_total_balanced_tarnet,
                               n_treated_balanced_tarnet):
        _train_parameters = self.__get_train_parameters_PM_GAN(tensor_treated_train)

        tensor_treated_test = \
            Utils.create_tensors_from_tuple(self.data_loader_dict_test["treated_data"])
        tensor_control_test = \
            Utils.create_tensors_from_tuple(self.data_loader_dict_test["control_data"])
        _test_parameters = self.__get_test_parameters(tensor_treated_test, tensor_control_test)

        tensor_treated_val = \
            Utils.create_tensors_from_tuple(self.data_loader_dict_val["treated_data"])
        tensor_control_val = \
            Utils.create_tensors_from_tuple(self.data_loader_dict_val["control_data"])
        _val_parameters = self.__get_test_parameters(tensor_treated_val, tensor_control_val)

        tarnet = TARNet_Manager(input_nodes=Constants.TARNET_INPUT_NODES,
                                shared_nodes=Constants.TARNET_SHARED_NODES,
                                outcome_nodes=Constants.TARNET_OUTPUT_NODES,
                                device=self.device)

        tarnet.train_semi_supervised(_train_parameters, _val_parameters,
                                     n_total_balanced_tarnet,
                                     n_treated_balanced_tarnet, self.device)
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
    def __get_train_parameters_ss(train_set):
        return {
            "epochs": 200,
            "lr": Constants.TARNET_LR,
            "lambda": Constants.TARNET_LAMBDA,
            "batch_size": 64,
            "shuffle": True,
            "tensor_dataset": train_set
        }

    @staticmethod
    def __get_train_parameters_PM_GAN(tensor_treated):
        return {
            "epochs": Constants.TARNET_EPOCHS,
            "lr": Constants.TARNET_LR,
            "lambda": Constants.TARNET_LAMBDA,
            "batch_size": Constants.TARNET_BATCH_SIZE,
            "shuffle": True,
            "tensor_dataset": tensor_treated
        }

    @staticmethod
    def __get_test_parameters(tensor_treated_test, tensor_control_test):
        return {
            "treated_set": tensor_treated_test,
            "control_set": tensor_control_test
        }
