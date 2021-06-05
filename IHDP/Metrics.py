import numpy as np
from Utils import Utils

class Metrics:
    @staticmethod
    def PEHE(y1_true_np, y0_true_np, y1_hat_np, y0_hat_np):
        # PEHE_val = np.sqrt(np.mean((
        #     (Utils.convert_to_col_vector(y1_true_np) - Utils.convert_to_col_vector(y0_true_np)) - (
        #                 Utils.convert_to_col_vector(y1_hat_np) - Utils.convert_to_col_vector(y0_hat_np))) ** 2))
        #
        PEHE_val = np.mean(np.abs((y1_true_np - y0_true_np) - (y1_hat_np - y0_hat_np)))
        return PEHE_val

    @staticmethod
    def ATE(y1_true_np, y0_true_np, y1_hat_np, y0_hat_np):
        ATE_val = np.abs(np.mean(y1_true_np - y0_true_np) - np.mean(y1_hat_np - y0_hat_np))
        return ATE_val
