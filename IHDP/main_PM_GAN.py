from Constants import Constants
from Experiments import Experiments
from Utils import Utils

if __name__ == '__main__':
    csv_path = "Dataset/ihdp_sample.csv"
    device = Utils.get_device()
    print(device)
    split_size = 0.8
    print("--->> !!Using NN prop score!! <<---")
    running_mode = "original_data"
    original_exp = Experiments(running_mode, csv_path, split_size)
    original_exp.run_all_experiments(iterations=Constants.ITERATIONS,
                                     ps_model_type=Constants.PS_MODEL_NN)
