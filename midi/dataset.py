import torch
from torch.utils.data import Dataset

from core.util import normalize
from midi.util import note_or_control_to_one_hot


class MidiDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        sequence = self.data[idx]
        encoded = []

        for item in sequence:
            one_hot = note_or_control_to_one_hot(item[0])
            encoded.append(torch.cat((
                one_hot,
                normalize(item[-2], max=127).reshape(1),
                normalize(item[-1], max=4295).reshape(1)
            )))
        ts = torch.stack(encoded)
        return ts[:-1], ts[-1]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def load(dataset_file):
        data = torch.load(dataset_file)

        return MidiDataset(data)
