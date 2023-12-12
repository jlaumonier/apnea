import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence
from torch.nn import LogSoftmax
import torch


# https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/
class BasicLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=10, num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(10, 1)
        self.sigmo = nn.Sigmoid()
        self.padding_value = -100.0

    def forward(self, input_):

        # print(input_.shape)
        # print(input_.dtype)

        lengths = [len(embeded_sequence) for embeded_sequence in input_]
        lengths = torch.Tensor(lengths)
        #print(lengths.shape)

        input_tensor = pad_sequence(input_,
                                    batch_first=True,
                                    padding_value=self.padding_value)

        # print(input_tensor.shape)
        # print(input_tensor.dtype)

        packed_input_tensor = pack_padded_sequence(input=input_tensor,
                                                   batch_first=True,
                                                   enforce_sorted=False,
                                                   lengths=lengths.cpu())

        lstm_output, _ = self.lstm(packed_input_tensor)

        lstm_out, _ = pad_packed_sequence(lstm_output,
                                          batch_first=True,
                                          padding_value=self.padding_value)

        #print(lstm_out)
        #print(lstm_out.shape)
        # print(lstm_out.dtype)

        output = self.fc1(lstm_out)

        #print(lin_output)
        # print(lin_output.shape)
        # print(lin_output.dtype)

        positive_output = self.sigmo(output)

        return positive_output
        #return output
