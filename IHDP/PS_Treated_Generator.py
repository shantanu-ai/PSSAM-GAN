import numpy as np
from matplotlib import pyplot

from Constants import Constants
from DCN_Experiments import DCN_Experiments
from GAN_Manager import GAN_Manager
from PSM_Manager import PSM_Manager
from TARNet_Experiments import TARNet_Experiments
from Utils import Utils
import seaborn as sns


class PS_Treated_Generator:
    def __init__(self, data_loader_dict_train, data_loader_dict_val, ps_model, ps_model_type):
        self.treated_tuple_full = data_loader_dict_train["treated_data"]
        self.control_tuple_full = data_loader_dict_train["control_data"]
        self.n_treated_original = data_loader_dict_train["treated_data"][0].shape[0]
        self.n_control_original = data_loader_dict_train["control_data"][0].shape[0]
        self.data_loader_dict_val = data_loader_dict_val
        self.ps_model = ps_model
        self.ps_model_type = ps_model_type

    def simulate_treated_semi_supervised(self, input_nodes, iter_id, device):
        treated_simulated, ps_treated_simulated, tuple_matched_control, tuple_unmatched_control \
            = self.__execute_GAN(device, iter_id)
        treated_simulated = treated_simulated.detach().cpu()

        treated_tensor_full_train = Utils.create_tensors_from_tuple(self.treated_tuple_full)
        control_tensor_full_train = Utils.create_tensors_from_tuple(self.control_tuple_full)
        ps_score_list_treated_np = np.array(ps_treated_simulated)

        tensor_treated_balanced_dcn, tensor_control_balanced_dcn, n_treated_balanced_dcn, \
        n_control_balanced_dcn = \
            self.__get_balanced_dataset_using_DCN(treated_simulated, ps_score_list_treated_np,
                                                  treated_tensor_full_train, control_tensor_full_train,
                                                  tuple_matched_control, tuple_unmatched_control,
                                                  input_nodes, device)
        tensor_balanced_tarnet, n_total, n_treated = \
            self.__get_balanced_dataset_using_TARNet(tuple_matched_control, tuple_unmatched_control,
                                                     treated_simulated,
                                                     ps_score_list_treated_np, input_nodes, device)

        return {
            "tensor_treated_balanced_dcn": tensor_treated_balanced_dcn,
            "tensor_control_balanced_dcn": tensor_control_balanced_dcn,
            "n_treated_balanced_dcn": n_treated_balanced_dcn,
            "n_control_balanced_dcn": n_control_balanced_dcn,

            "tensor_balanced_tarnet": tensor_balanced_tarnet,
            "n_total_balanced_tarnet": n_total,
            "n_treated_balanced_tarnet": n_treated
        }

    def __get_balanced_dataset_using_TARNet(self,
                                            tuple_matched_control, tuple_unmatched_control,
                                            treated_simulated,
                                            ps_score_list_treated_np, input_nodes, device):
        np_treated_x, np_treated_ps, np_treated_f, np_treated_cf = self.treated_tuple_full
        np_control_x, np_control_ps, np_control_f, np_control_cf = self.control_tuple_full

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

        train_set = Utils.create_tensors_to_train_DCN_semi_supervised(
            (np_train_ss_X, np_train_ss_ps, np_train_ss_T, np_train_ss_f,
             np_train_ss_cf))
        eval_set = Utils.convert_to_tensor_DCN_PS(treated_simulated,
                                                  ps_score_list_treated_np)
        print("--" * 20)
        print("----->>> Semi supervised training started for TARNet <<<-----")

        tarnet = TARNet_Experiments(input_nodes, device)
        simulated_treated_Y = tarnet.semi_supervised_train_eval(train_set, self.data_loader_dict_val,
                                                                eval_set, n_total, n_treated)

        print("----->>> Semi supervised training completed <<<-----")

        np_treated_gen_f = Utils.convert_to_col_vector(simulated_treated_Y["y_f_list"])
        np_treated_gen_cf = Utils.convert_to_col_vector(simulated_treated_Y["y_cf_list"])
        np_treated_x, np_treated_ps, np_treated_f, np_treated_cf = \
            self.__get_balanced_treated(treated_simulated,
                                        ps_score_list_treated_np,
                                        np_treated_gen_f,
                                        np_treated_gen_cf)

        t_1 = np.ones(np_treated_x.shape[0])
        t_0 = np.zeros(np_control_x.shape[0])
        np_train_supervised_X = np.concatenate((np_treated_x, np_control_x), axis=0)
        np_train_supervised_ps = np.concatenate((np_treated_ps, np_control_ps), axis=0)
        np_train_supervised_T = np.concatenate((t_1, t_0), axis=0)
        np_train_supervised_f = np.concatenate((np_treated_f, np_control_f), axis=0)
        np_train_supervised_cf = np.concatenate((np_treated_cf, np_control_cf), axis=0)

        print("TARnet Supervised Model dataset statistics:")
        print(np_treated_x.shape)
        print(np_control_x.shape)
        print(np_train_supervised_X.shape)

        n_treated = np_treated_x.shape[0]
        n_control = np_control_x.shape[0]
        n_total = n_treated + n_control

        tensor_balanced = Utils.create_tensors_to_train_DCN_semi_supervised(
            (np_train_supervised_X, np_train_supervised_ps, np_train_supervised_T,
             np_train_supervised_f,
             np_train_supervised_cf))

        # np_control_x, np_control_ps, np_control_f, np_control_cf \
        #     = self.__get_balanced_control(tuple_matched_control,
        #                                   tuple_unmatched_control)
        # tuple_control_balanced = (np_control_x, np_control_ps, np_control_f, np_control_cf)
        return tensor_balanced, n_total, n_treated

    def __get_balanced_dataset_using_DCN(self, treated_simulated, ps_score_list_treated_np,
                                         treated_tensor_full_train, control_tensor_full_train,
                                         tuple_matched_control, tuple_unmatched_control,
                                         input_nodes, device):
        eval_set = Utils.convert_to_tensor_DCN_PS(treated_simulated,
                                                  ps_score_list_treated_np)
        print("--" * 20)
        print("----->>> Semi supervised training started for DCN <<<-----")

        dcn = DCN_Experiments(input_nodes, device)
        simulated_treated_Y = dcn.semi_supervised_train_eval(treated_tensor_full_train,
                                                             control_tensor_full_train,
                                                             self.data_loader_dict_val,
                                                             self.n_treated_original,
                                                             self.n_control_original,
                                                             eval_set)
        print("---> Semi supervised training completed...")
        np_treated_gen_f = Utils.convert_to_col_vector(simulated_treated_Y["y_f_list"])
        np_treated_gen_cf = Utils.convert_to_col_vector(simulated_treated_Y["y_cf_list"])
        np_treated_x, np_treated_ps, np_treated_f, np_treated_cf = \
            self.__get_balanced_treated(treated_simulated,
                                        ps_score_list_treated_np,
                                        np_treated_gen_f,
                                        np_treated_gen_cf)

        np_control_x, np_control_ps, np_control_f, np_control_cf = \
            self.__get_balanced_control(tuple_matched_control,
                                        tuple_unmatched_control)

        tensor_treated_balanced = Utils.convert_to_tensor_DCN(np_treated_x, np_treated_ps,
                                                              np_treated_f, np_treated_cf)

        tensor_control_balanced = Utils.convert_to_tensor_DCN(np_control_x, np_control_ps,
                                                              np_control_f, np_control_cf)

        n_treated_balanced = np_treated_x.shape[0]
        n_control_balanced = np_control_x.shape[0]

        return tensor_treated_balanced, tensor_control_balanced, n_treated_balanced, n_control_balanced

    @staticmethod
    def __get_balanced_control(tuple_matched_control, tuple_unmatched_control):
        np_control_unmatched_X = tuple_unmatched_control[0]
        np_ps_score_list_control_unmatched = tuple_unmatched_control[1]
        np_control_unmatched_f = tuple_unmatched_control[2]
        np_control_unmatched_cf = tuple_unmatched_control[3]

        np_control_matched_X = tuple_matched_control[0]
        np_ps_score_list_control_matched = tuple_matched_control[1]
        np_control_matched_f = tuple_matched_control[2]
        np_control_matched_cf = tuple_matched_control[3]

        np_control_x = np.concatenate((np_control_unmatched_X, np_control_matched_X), axis=0)
        np_control_ps = np.concatenate((np_ps_score_list_control_unmatched,
                                        np_ps_score_list_control_matched), axis=0)
        np_control_f = np.concatenate((np_control_unmatched_f, np_control_matched_f), axis=0)
        np_control_cf = np.concatenate((np_control_unmatched_cf, np_control_matched_cf), axis=0)

        return np_control_x, np_control_ps, np_control_f, np_control_cf

    def __get_balanced_treated(self, treated_simulated, ps_score_list_treated_np,
                               np_treated_gen_f, np_treated_gen_cf):
        np_treated_generated = treated_simulated.numpy()

        np_original_X = self.treated_tuple_full[0]
        np_original_ps_score = self.treated_tuple_full[1]
        np_original_Y_f = self.treated_tuple_full[2]
        np_original_Y_cf = self.treated_tuple_full[3]

        np_treated_x = np.concatenate((np_treated_generated, np_original_X), axis=0)
        np_treated_ps = np.concatenate((ps_score_list_treated_np, np_original_ps_score), axis=0)
        np_treated_f = np.concatenate((np_treated_gen_f, np_original_Y_f), axis=0)
        np_treated_cf = np.concatenate((np_treated_gen_cf, np_original_Y_cf), axis=0)

        return np_treated_x, np_treated_ps, np_treated_f, np_treated_cf

    def __execute_GAN(self, device, iter_id):
        psm = PSM_Manager()
        control_set = psm.match_using_prop_score(self.treated_tuple_full,
                                                 self.control_tuple_full)
        tuple_matched_control = control_set["tuple_matched_control"]
        tuple_unmatched_control = control_set["tuple_unmatched_control"]

        print("-> Matched Control: {0}".format(tuple_matched_control[0].shape))
        print("-> UnMatched Control: {0}".format(tuple_unmatched_control[0].shape))
        print("-> GAN training started")

        tensor_unmatched_control = \
            Utils.create_tensors_from_tuple(tuple_unmatched_control)

        GAN_train_parameters = {
            "epochs": Constants.GAN_EPOCHS,
            "lr": Constants.GAN_LR,
            "shuffle": True,
            "train_set": tensor_unmatched_control,
            "batch_size": Constants.GAN_BATCH_SIZE,
            "BETA": Constants.GAN_BETA
        }

        gan = GAN_Manager(Constants.GAN_DISCRIMINATOR_IN_NODES,
                          Constants.GAN_GENERATOR_OUT_NODES,
                          self.ps_model, self.ps_model_type, device)
        gan.train_GAN(GAN_train_parameters, device=device)
        print("-> GAN training completed")
        treated_generated, ps_score_list_sim_treated = gan.eval_GAN(tuple_unmatched_control[0].shape[0],
                                                                    device)

        ps_matched_control_list = tuple_matched_control[1].tolist()
        ps_un_matched_control_list = tuple_unmatched_control[1].tolist()
        ps_treated_list = self.treated_tuple_full[1].tolist()
        ps_control_list = self.control_tuple_full[1].tolist()

        # matched control and treated
        self.draw(ps_treated_list, ps_matched_control_list,
                  label_treated="Matched treated", label_control="Matched control",
                  fig_name="./Plots/Fig_Iter_id_{0}_Matched treated vs Matched Control".format(iter_id),
                  title="IHDP: PSM dataset")

        # full control and treated
        self.draw(ps_treated_list, ps_control_list,
                  label_treated="Matched treated", label_control="Unmatched control",
                  fig_name="./Plots/Fig_Iter_id_{0}_Matched treated vs Unmatched Control".format(iter_id),
                  title="IHDP: original dataset")

        # treated by GAN vs unmatched control
        self.draw(ps_score_list_sim_treated, ps_un_matched_control_list,
                  label_treated="Synthetic treated", label_control="Unmatched control",
                  fig_name="./Plots/Fig_Iter_id_{0}_Simulated treated vs Unmatched Control".format(iter_id),
                  title="IHDP: original + GAN dataset")

        return treated_generated, ps_score_list_sim_treated, tuple_matched_control, tuple_unmatched_control

    @staticmethod
    def draw(treated_ps_list, control_ps_list, label_treated, label_control, fig_name, title):
        bins1 = np.linspace(0, 1, 100)
        pyplot.hist(treated_ps_list, bins1, alpha=0.5, label=label_treated)
        pyplot.hist(control_ps_list, bins1, alpha=0.5, label=label_control)
        pyplot.xlabel('Propensity scores', fontsize=10)
        pyplot.ylabel('Frequency', fontsize=10)
        pyplot.title(title)
        pyplot.xticks(fontsize=7)
        pyplot.yticks(fontsize=7)
        pyplot.legend(loc='upper right')
        pyplot.draw()
        pyplot.savefig(fig_name)

        # pyplot.show()
        pyplot.clf()
