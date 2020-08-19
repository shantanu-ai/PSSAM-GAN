import numpy as np


class Metrics:
    @staticmethod
    def PEHE(y1_true_np, y0_true_np, y1_hat_np, y0_hat_np):
        PEHE_val = np.mean(np.abs((y1_true_np - y0_true_np) - (y1_hat_np - y0_hat_np)))
        return PEHE_val

    def PEHE_new(y1_true_np, y0_true_np, y1_hat_np, y0_hat_np):
        PEHE_val = np.sqrt(np.mean(np.square((y1_true_np - y0_true_np) - (y1_hat_np - y0_hat_np))))
        return PEHE_val

    @staticmethod
    def ATE(y1_true_np, y0_true_np, y1_hat_np, y0_hat_np):
        ATE_val = np.abs(np.mean(y1_true_np - y0_true_np) - np.mean(y1_hat_np - y0_hat_np))
        return ATE_val
