import torch


def prediction_matrix_to_metrics(matrix):
    return average_accuracy(matrix), average_precision(matrix), average_recall(matrix)


def average_accuracy(matrix):
    return torch.mean(torch.sum(torch.diagonal(matrix, 0)) / torch.sum(matrix)).item()


def average_precision(matrix):
    return torch.mean(torch.diagonal(matrix, 0) / torch.max(torch.ones(len(matrix)), torch.sum(matrix, dim=1))).item()


def average_recall(matrix):
    return torch.mean(torch.diagonal(matrix, 0) / torch.max(torch.ones(len(matrix)), torch.sum(matrix, dim=0))).item()
