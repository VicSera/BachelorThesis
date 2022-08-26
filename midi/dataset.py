import torch
from torch.utils.data import Dataset

from midi.util import sequence_to_input


class MidiDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        sequence = self.data[idx]

        pitch, normalized = sequence_to_input(sequence[:-1])
        return pitch, normalized, sequence[-1, 0].long(), sequence[-1, 1:]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def load(dataset_file):
        data = torch.load(dataset_file)

        return MidiDataset(data)
