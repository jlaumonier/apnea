import torch
from torch.nn import LogSoftmax
from poutyne.framework.metrics import acc


def accuracy(pred: torch.Tensor,
             ground_truth: torch.Tensor,
             device: torch.device = 'cpu',
             ignore_idx: int = -100) -> int:
    if isinstance(pred, tuple):
        pred = pred[0]

    activation = LogSoftmax(dim=2)
    pred = activation(pred)

    y_true = ground_truth.type(torch.LongTensor).to(device)
    y_pred = pred.to(device)

    weights = (y_true != ignore_idx).float()
    num_labels = weights.sum()
    acc_pred = (y_pred == y_true).float() * weights

    acc_pred = acc_pred / num_labels

    return acc_pred * 100

