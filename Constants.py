class Constants:
    # ps score neural net training parameters
    PROP_SCORE_NN_EPOCHS = 50
    PROP_SCORE_NN_LR = 0.001
    PROP_SCORE_NN_BATCH_SIZE = 32
    PROP_SCORE_NN_MODEL_PATH = "./Propensity_Model/NN_PS_model_iter_id_{0}_epoch_{1}_lr_{2}.pth"

    # DCN training parameters
    DCN_EPOCHS = 100
    DCN_LR = 0.001
    DCN_MODEL_PATH_PD_PM_MATCH_FALSE = "./DCNModel/NN_DCN_PD_PM_MATCH_FALSE_model_iter_id_{0}_epoch_{1}_lr_{2}.pth"
    DCN_MODEL_PATH_PD_PM_MATCH_TRUE = "./DCNModel/NN_DCN_PD_PM_MATCH_TRUE_model_iter_id_{0}_epoch_{1}_lr_{2}.pth"

    DCN_MODEL_PATH_CONSTANT_DROPOUT_5 = "./DCNModel/NN_DCN_CONSTANT_DROPOUT_5_model_iter_id_{0}_epoch_{1}_lr_{2}.pth"
    DCN_MODEL_PATH_CONSTANT_DROPOUT_2 = "./DCNModel/NN_DCN_CONSTANT_DROPOUT_2_model_iter_id_{0}_epoch_{1}_lr_{2}.pth"

    DCN_MODEL_SEMI_SUPERVISED_PATH = "./DCNModel/NN_DCN_SEMI_SUPERVISED_NO_DROPOUT_PM_MATCH_FALSE_model_iter_id_{0}_epoch_{1}_lr_{2}.pth"
    DCN_MODEL_PATH_NO_DROPOUT_PM_MATCH_FALSE = "./DCNModel/NN_DCN_NO_DROPOUT_PM_MATCH_FALSE_model_iter_id_{0}_epoch_{1}_lr_{2}.pth"

    DCN_MODEL_PATH_NO_DROPOUT_PM_MATCH_TRUE = "./DCNModel/NN_DCN_NO_DROPOUT_PM_MATCH_TRUE_model_iter_id_{0}_epoch_{1}_lr_{2}.pth"

    # train DCN types
    DCN_EVALUATION = "eval"
    DCN_TRAIN_CONSTANT_DROPOUT_5 = "train_constant_dropout_5"
    DCN_TRAIN_CONSTANT_DROPOUT_2 = "train_constant_dropout_2"
    DCN_TRAIN_PD = "train_PD"
    DCN_TRAIN_NO_DROPOUT = "train_with_no_dropout"