from sklearn.linear_model import LogisticRegression

from Constants import Constants
from Propensity_socre_network import Propensity_socre_network


class PS_Manager:
    @staticmethod
    def get_propensity_scores(ps_train_set, iter_id, input_nodes, device):
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

        print(ps_score_list_train)
        return ps_score_list_train, ps_net_NN

    @staticmethod
    def get_propensity_scores_using_LR(np_covariates_X_train, np_covariates_Y_train,
                                       np_covariates_X_val,
                                       np_covariates_X_test,
                                       regularized=False):
        print("############### Propensity Score Logistic Regression Training ###############")

        ps_model = None
        if not regularized:
            ps_model = LogisticRegression(solver='liblinear')
        elif regularized:
            ps_model = LogisticRegression(penalty="l1", solver="liblinear")
        np_covariates_X_train = np_covariates_X_train[:, :-2]
        np_covariates_X_test = np_covariates_X_test[:, :-2]

        ps_model.fit(np_covariates_X_train, np_covariates_Y_train.ravel())
        ps_score_list_train = ps_model.predict_proba(np_covariates_X_train)[:, -1].tolist()
        ps_score_list_val = ps_model.predict_proba(np_covariates_X_val)[:, -1].tolist()
        ps_score_list_test = ps_model.predict_proba(np_covariates_X_test)[:, -1].tolist()

        return ps_score_list_train, ps_score_list_val, ps_score_list_test, ps_model

    @staticmethod
    def get_propensity_scores_using_LR_GAN(np_covariates_X_test, np_covariates_Y_test, log_reg):
        # fit the model with data
        proba = log_reg.predict_proba(np_covariates_X_test)[:, -1].tolist()
        return proba
