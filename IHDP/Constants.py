class Constants:
    # ps score neural net training parameters
    PROP_SCORE_NN_EPOCHS = 50
    PROP_SCORE_NN_LR = 0.001
    PROP_SCORE_NN_BATCH_SIZE = 32
    PROP_SCORE_NN_MODEL_PATH = "./Propensity_Model/NN_PS_model_iter_id_{0}_epoch_{1}_lr_{2}.pth"

    # DCN training parameters
    DCN_EPOCHS = 1
    DCN_LR = 0.0001
    DCN_MODEL_PATH_PD_PM_MATCH_FALSE = "./DCNModel/NN_DCN_PD_PM_MATCH_FALSE_model_iter_id_{0}_epoch_{1}_lr_{2}.pth"
    DCN_MODEL_PATH_PD_PM_MATCH_TRUE = "./DCNModel/NN_DCN_PD_PM_MATCH_TRUE_model_iter_id_{0}_epoch_{1}_lr_{2}.pth"

    DCN_MODEL_PATH_CONSTANT_DROPOUT_5 = "./DCNModel/NN_DCN_CONSTANT_DROPOUT_5_model_iter_id_{0}_epoch_{1}_lr_{2}.pth"
    DCN_MODEL_PATH_CONSTANT_DROPOUT_2 = "./DCNModel/NN_DCN_CONSTANT_DROPOUT_2_model_iter_id_{0}_epoch_{1}_lr_{2}.pth"

    DCN_MODEL_SEMI_SUPERVISED_PATH = "./DCNModel/NN_DCN_SEMI_SUPERVISED_NO_DROPOUT_PM_MATCH_FALSE_model_iter_id_{0}_epoch_{1}_lr_{2}.pth"
    DCN_MODEL_PATH_NO_DROPOUT_PM_MATCH_FALSE = "./DCNModel/NN_DCN_NO_DROPOUT_PM_MATCH_FALSE_model_iter_id_{0}_epoch_{1}_lr_{2}.pth"

    DCN_MODEL_PATH_NO_DROPOUT_PM_MATCH_TRUE = "./DCNModel/NN_DCN_NO_DROPOUT_PM_MATCH_TRUE_model_iter_id_{0}_epoch_{1}_lr_{2}.pth"

    # train modes DCN
    DCN_EVALUATION = "eval"
    DCN_TRAIN_CONSTANT_DROPOUT_5 = "train_constant_dropout_5"
    DCN_TRAIN_CONSTANT_DROPOUT_2 = "train_constant_dropout_2"
    DCN_TRAIN_PD = "train_PD"
    DCN_TRAIN_NO_DROPOUT = "train_with_no_dropout"

    # ps model types
    PS_MODEL_NN = "Neural_Net"
    PS_MODEL_LR = "Logistic"
    PS_MODEL_LR_Lasso = "Logistic_L1"

    # GAN Training Ihdp
    GAN_EPOCHS = 1
    GAN_LR = 0.0002
    GAN_BETA = 1
    GAN_BATCH_SIZE = 64
    GAN_GENERATOR_IN_NODES = 25
    # GAN_GENERATOR_IN_NODES = 200
    GAN_GENERATOR_OUT_NODES = 25
    GAN_DISCRIMINATOR_IN_NODES = 25

    # TARNet
    TARNET_EPOCHS = 1
    TARNET_SS_EPOCHS = 1
    TARNET_LR = 1e-3
    TARNET_LAMBDA = 1e-4
    TARNET_BATCH_SIZE = 100
    TARNET_INPUT_NODES = 25
    TARNET_SHARED_NODES = 200
    TARNET_OUTPUT_NODES = 100
