import torch.nn.functional as F


def custom_two_part_loss(target, actual):
    loss1 = F.cross_entropy(input=actual[0],
                            target=target[:, :-2])
    loss2 = F.mse_loss(input=actual[1],
                       target=target[:, -2:])

    return loss1 + loss2
