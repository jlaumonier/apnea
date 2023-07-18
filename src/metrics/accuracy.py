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

    result = acc(pred.to(device),
                 ground_truth.type(torch.LongTensor).to(device),
                 ignore_index=ignore_idx)
    return result
