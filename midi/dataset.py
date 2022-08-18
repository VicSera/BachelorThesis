import torch
from torch.utils.data import Dataset

from core.config import Config


class MidiDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        sequence = self.data[idx]

        normalized = torch.div(sequence[:-1], torch.tensor([Config.NOTES_COUNT, 1, 1, 1]))
        return normalized, sequence[-1, 0].long(), sequence[-1, 1:]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def load(dataset_file):
        data = torch.load(dataset_file)

        return MidiDataset(data)
