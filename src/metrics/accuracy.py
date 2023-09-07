import torch

from poutyne.framework.metrics import acc


def accuracy(pred: torch.Tensor,
             ground_truth: torch.Tensor,
             device: torch.device = 'cpu',
             ignore_idx: int = -100,
             reduction='mean') -> int:
    if isinstance(pred, tuple):
        pred = pred[0]




    y_true = ground_truth.type(torch.LongTensor).to(device)
    y_pred = pred.to(device)

    # From Poutyne - Maybe problems with Licencing. TODO rewrite!

    weights = (y_true != ignore_idx).float()
    num_labels = weights.sum()
    acc_pred = (y_pred == y_true).float() * weights

    if reduction in ['mean', 'sum']:
        acc_pred = acc_pred.sum()

    if reduction == 'mean':
        acc_pred = acc_pred / num_labels

    return acc_pred * 100
    # From Poutyne

