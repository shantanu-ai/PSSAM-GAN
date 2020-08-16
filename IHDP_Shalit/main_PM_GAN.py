from Constants import Constants
from Experiments import Experiments

if __name__ == '__main__':
    train_path = "Dataset/ihdp_npci_1-100.train.npz"
    test_path = "Dataset/ihdp_npci_1-100.test.npz"
    print("Using original data")
    running_mode = "original_data"
    original_exp = Experiments(running_mode)
    original_exp.run_all_experiments(train_path, test_path,
                                     iterations=1,
                                     ps_model_type=Constants.PS_MODEL_NN)
