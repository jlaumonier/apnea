import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence
from torch.nn import LogSoftmax


# https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/
class BasicLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=100, num_layers=10, batch_first=True)
        self.linear = nn.Linear(100, 1)
        self.sftmax = LogSoftmax(dim=2)

    def forward(self, input_):
        # todo packed_pad_seq au lieu du collator

        lstm_output, _ = self.lstm(input_)

        lstm_out, _ = pad_packed_sequence(lstm_output, batch_first=True, padding_value=-100.0)

        lin_output = self.linear(lstm_out)

        output = self.sftmax(lin_output)

        return output
