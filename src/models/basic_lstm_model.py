import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence, PackedSequence

# https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/
class BasicLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, input_):
        lstm_output, _ = self.lstm(input_)

        lstm_out, _ = pad_packed_sequence(lstm_output, batch_first=True)

        output = self.linear(lstm_out)

        return output