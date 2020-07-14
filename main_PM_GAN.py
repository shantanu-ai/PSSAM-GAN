# from Graphs import Graphs
from Experiments import Experiments
# from DCN_PD_test import DCN_PD_Deep

if __name__ == '__main__':
    print("Using original data")
    Experiments().run_all_experiments(iterations=1,
                                      running_mode="original_data")
