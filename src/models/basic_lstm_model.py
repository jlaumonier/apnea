import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn import LogSoftmax
import torch


# https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/
class BasicLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=100, num_layers=10, batch_first=True)
        self.linear = nn.Linear(100, 1)
        self.sftmax = LogSoftmax(dim=2)

    def forward(self, input_):
        # todo packed_pad_seq au lieu du collator

        print(input_.shape)

        l = [(torch.Tensor(embeded_sequence), len(embeded_sequence)) for embeded_sequence in input_]
        print(l)
        input_tensor, lengths = zip(*l)

        print(input_tensor)

        packed_input_tensor = pack_padded_sequence(input_tensor, batch_first=True,
                                                   enforce_sorted=False,
                                                   lengths=lengths)

        #print(packed_input_tensor.shape)

        lstm_output, _ = self.lstm(packed_input_tensor)

        print(lstm_output.shape)

        lstm_out, _ = pad_packed_sequence(lstm_output, batch_first=True, padding_value=-100.0)

        print(lstm_out.shape)

        lin_output = self.linear(lstm_out)

        print(lin_output.shape)

        output = self.sftmax(lin_output)

        print(output.shape)

        return output
