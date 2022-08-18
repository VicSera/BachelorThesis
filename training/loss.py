import torch.nn.functional as F

from core.config import Config


def custom_two_part_loss(pitch, extra, pitch_pred, extra_pred):
    cross_entropy = F.cross_entropy(input=pitch_pred,
                                    target=pitch)
    mse = F.mse_loss(input=extra_pred,
                     target=extra)

    weighted = (cross_entropy * Config.CROSS_ENTROPY_WEIGHT + mse * Config.MSE_WEIGHT) / Config.WEIGHT_TOTAL

    return cross_entropy, mse, weighted
