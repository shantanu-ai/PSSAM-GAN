import numpy as np
from matplotlib import pyplot

from Constants import Constants
from GAN_Manager import GAN_Manager
from PSM_Manager import PSM_Manager
from Utils import Utils


class PS_Treated_Generator:
    def __init__(self, data_loader_dict_train, ps_model, ps_model_type):
        self.treated_tuple_full = data_loader_dict_train["treated_data"]
        self.control_tuple_full = data_loader_dict_train["control_data"]
        self.n_treated_original = data_loader_dict_train["treated_data"][0].shape[0]
        self.n_control_original = data_loader_dict_train["control_data"][0].shape[0]
        self.ps_model = ps_model
        self.ps_model_type = ps_model_type

    def simulate_treated_semi_supervised(self, input_nodes, iter_id, device):
        treated_simulated, ps_treated_simulated, tuple_matched_control, tuple_unmatched_control \
            = self.__execute_GAN(device, iter_id)

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
                  label_treated="Treatment Resist", label_control="Treatment Suscept",
                  fig_name="./Plots/Fig_Iter_id_{0}_Matched".format(iter_id),
                  title="Methicilin: PSM dataset", max_limit=200)

        # full control and treated
        self.draw(ps_treated_list, ps_control_list,
                  label_treated="Treatment Resist", label_control="Treatment Suscept",
                  fig_name="./Plots/Fig_Iter_id_{0}_Original".format(iter_id),
                  title="Methicilin: original dataset", max_limit=200)

        # treated by GAN vs unmatched control
        self.draw(ps_score_list_sim_treated + ps_treated_list, ps_control_list,
                  label_treated="Treatment Resist", label_control="Treatment Suscept",
                  fig_name="./Plots/Fig_Iter_id_{0}_Simulated".format(iter_id),
                  title="Methicilin: original + GAN dataset", max_limit=200)

        return treated_generated, ps_score_list_sim_treated, tuple_matched_control, tuple_unmatched_control

    @staticmethod
    def draw(treated_ps_list, control_ps_list, label_treated, label_control, fig_name, title, max_limit):
        bins1 = np.linspace(0, 1, 50)
        max_T = max(treated_ps_list)
        max_C = max(control_ps_list)
        max_X = max(max_T, max_C)
        pyplot.hist(treated_ps_list, bins1, alpha=0.5, label=label_treated, color='#B60E0E', histtype="bar",
                    edgecolor='r')
        pyplot.hist(control_ps_list, bins1, alpha=0.5, label=label_control, color='g', histtype="bar",
                    edgecolor='g')
        pyplot.xlabel('Propensity scores', fontsize=12)
        pyplot.ylabel('Frequency', fontsize=12)
        pyplot.title(title)
        pyplot.ylim(0, max_limit)
        # pyplot.xlim(0, max_X)
        pyplot.xticks(fontsize=7)
        pyplot.yticks(fontsize=7)
        pyplot.legend(loc='upper right')
        pyplot.draw()
        pyplot.savefig(fig_name, dpi=220)
        pyplot.clf()
