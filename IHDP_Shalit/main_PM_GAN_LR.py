from Constants import Constants
from Experiments import Experiments

if __name__ == '__main__':
    train_path = "Dataset/ihdp_npci_1-100.train.npz"
    test_path = "Dataset/ihdp_npci_1-100.test.npz"

    print("--->> !!Using LR prop score!! <<---")
    running_mode = "original_data"
    original_exp = Experiments(running_mode)
    original_exp.run_all_experiments(train_path, test_path,
                                     iterations=100,
                                     ps_model_type=Constants.PS_MODEL_LR)
