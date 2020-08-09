from Constants import Constants
from DCN_Manager import DCN_Manager
from Utils import Utils


class DCN_Experiments:
    def __init__(self, data_loader_dict_train, data_loader_dict_test, input_nodes, device):
        # data loader -> (np_treated_df_X, np_treated_ps_score, np_treated_df_Y_f, np_treated_df_Y_cf)

        self.data_loader_dict_train = data_loader_dict_train
        self.data_loader_dict_test = data_loader_dict_test
        self.input_nodes = input_nodes
        self.device = device

    def evaluate_DCN_Model(self):
        # Model 1: DCN - PD
        dcn_pd_eval_dict = self.evaluate_DCN_PD()
        # Model 2: PM GAN - No drop0ut

        # Model 3: PM GAN - dropout - 0.2

        # Model 4: PM GAN - dropout - 0.5

        return {
            "dcn_pd_eval_dict": dcn_pd_eval_dict
        }

    def evaluate_DCN_PD(self):
        DCN_train_parameters = self.__get_train_parameters()
        train_mode = Constants.DCN_TRAIN_PD
        DCN_test_parameters = self.__get_test_parameters()
        return self.__supervised_train_eval(train_mode, DCN_train_parameters,
                                            DCN_test_parameters)

    def __supervised_train_eval(self, train_mode, DCN_train_parameters,
                                DCN_test_parameters):
        dcn_pd = DCN_Manager(self.input_nodes, self.device)
        dcn_pd.train(DCN_train_parameters, self.device, train_mode=train_mode)
        dcn_eval_dict = dcn_pd.eval(DCN_test_parameters, self.device)
        return dcn_eval_dict

    def __get_train_parameters(self):
        tensor_treated_train = \
            Utils.create_tensors_from_tuple(self.data_loader_dict_train["treated_data"])
        tensor_control_train = \
            Utils.create_tensors_from_tuple(self.data_loader_dict_train["control_data"])
        return {
            "epochs": Constants.DCN_EPOCHS,
            "lr": Constants.DCN_LR,
            "treated_batch_size": 1,
            "control_batch_size": 1,
            "shuffle": True,
            "treated_set_train": tensor_treated_train,
            "control_set_train": tensor_control_train
        }

    def __get_test_parameters(self):
        tensor_treated_test = \
            Utils.create_tensors_from_tuple(self.data_loader_dict_test["treated_data"])
        tensor_control_test = \
            Utils.create_tensors_from_tuple(self.data_loader_dict_test["control_data"])
        return {
            "treated_set": tensor_treated_test,
            "control_set": tensor_control_test
        }
