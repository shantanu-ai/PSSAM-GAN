from Constants import Constants
from Propensity_socre_network import Propensity_socre_network


class PS_Manager:
    @staticmethod
    def get_propensity_scores(ps_train_set, ps_test_set, iter_id, input_nodes, device):
        print("############### Propensity Score neural net Training ###############")

        train_parameters_NN = {
            "epochs": Constants.PROP_SCORE_NN_EPOCHS,
            "lr": Constants.PROP_SCORE_NN_LR,
            "batch_size": Constants.PROP_SCORE_NN_BATCH_SIZE,
            "shuffle": True,
            "train_set": ps_train_set
        }
        ps_net_NN = Propensity_socre_network(input_nodes, device)
        ps_net_NN.set_train_mode(phase="train")
        ps_net_NN.train(train_parameters_NN, device)

        # eval
        eval_parameters_train_NN = {
            "eval_set": ps_train_set
        }
        ps_net_NN.set_train_mode(phase="eval")
        ps_score_list_train = ps_net_NN.eval(eval_parameters_train_NN, device)

        ps_eval_parameters_NN = {
            "eval_set": ps_test_set
        }
        ps_score_list_test = ps_net_NN.eval(ps_eval_parameters_NN, device)

        return ps_score_list_train, ps_score_list_test
