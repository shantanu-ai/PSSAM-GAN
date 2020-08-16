from Constants import Constants
from Experiments import Experiments

if __name__ == '__main__':
    csv_path = "Dataset/ihdp_sample.csv"
    split_size = 0.8

    print("--->> !!Using LR prop score!! <<---")
    running_mode = "original_data"
    original_exp = Experiments(running_mode, csv_path, split_size)
    original_exp.run_all_experiments(iterations=1,
                                     ps_model_type=Constants.PS_MODEL_LR)
