import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence


# https://github.com/iid-ulaval/CCF-dataset/blob/main/experiment/src/data/collator/embedding_collator.py
class TSCollator:
    """
    Time serie collator
    """

    def __init__(self,
                 padding_value: int = 0,
                 ):
        self.padding_value = padding_value

    def collate_batch(
            self, batch):

        input_tensor, target_tensor, lengths = zip(*[(torch.Tensor(embeded_sequence), torch.Tensor(target),
                                                      len(embeded_sequence)
                                                      ) for (embeded_sequence, target) in  batch])

        print(input_tensor[0].shape)
        print(target_tensor[0].shape)
        print(len(lengths))

        lengths = torch.Tensor(lengths)

        input_tensor = pad_sequence(input_tensor,
                                    batch_first=True,
                                    padding_value=self.padding_value)
        print(input_tensor.shape)

        pack_padded_sequences_vectors = pack_padded_sequence(input_tensor, lengths.cpu(), batch_first=True)

        print(pack_padded_sequences_vectors.shape)

        target_tensor = pad_sequence(target_tensor,
                                     batch_first=True,
                                     padding_value=-100)

        print(target_tensor.shape)

        return pack_padded_sequences_vectors, target_tensor
