import os

import torchaudio
from torch.utils.data import Dataset
import torch
from torchaudio.transforms import Resample, MelSpectrogram
from tqdm import tqdm

from core.config import Config
from core.util import format_input_output
from model.util import quantize_data
from training.audio_utils import to_single_channel
from training.util import window


def quantize_tensor(x):
    return torch.from_numpy(quantize_data(x, 256))


def one_hot(x):
    enc = torch.FloatTensor(256, Config.in_seq_len).zero_()
    x = x.type('torch.LongTensor')
    enc.scatter_(0, x.unsqueeze(0), 1.)
    return enc


class DrumDataset(Dataset):
    def __init__(self, X, Y, Cond):
        self.X = X
        self.Y = Y
        self.Cond = Cond

    def __getitem__(self, item):
        return one_hot(self.X[item]), one_hot(self.Cond[item]), self.Y[item].unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def save(self, output_dir):
        print("Saving dataset to", output_dir)
        torch.save(self.X, f'{output_dir}/X.pt')
        torch.save(self.Cond, f'{output_dir}/Cond.pt')
        torch.save(self.Y, f'{output_dir}/Y.pt')

    @staticmethod
    def load(output_dir):
        X = torch.load(f'{output_dir}/X.pt')
        Cond = torch.load(f'{output_dir}/Cond.pt')
        Y = torch.load(f'{output_dir}/Y.pt')

        return DrumDataset(X=X, Cond=Cond, Y=Y)


def preprocess_track(track, resampler):
    return quantize_tensor(to_single_channel(resampler(track)))


def get_dataset():
    X, Cond, Y = [], [], []

    songs = [format_input_output(directory) for directory in os.walk(Config.input_directory)]
    print("Creating batches...")
    for song in songs:
        if 'drums' in song.keys():
            print(f"Song: {song['directory']}")
            # condition_track1, sr = torchaudio.load(song['bass'], normalize=True)
            condition_track2, sr = torchaudio.load(song['other'], normalize=True)
            # condition_track3, _ = torchaudio.load(song['vocals'], normalize=True)
            target_track, _ = torchaudio.load(song['drums'], normalize=True)

            # Preprocess
            resampler = Resample(sr, Config.sample_rate)

            target_track = preprocess_track(target_track, resampler)
            # condition_track1 = preprocess_track(condition_track1, resampler)
            condition_track2 = preprocess_track(condition_track2, resampler)
            # condition_track3 = preprocess_track(condition_track3, resampler)

            length = target_track.size()[-1]  # number of samples in the whole song

            num_sequences = length // Config.in_seq_len

            for seq_number in tqdm(range(num_sequences)):
                seq_head = seq_number * Config.in_seq_len

                x = window(target_track, head=seq_head, lookback=Config.in_seq_len)
                # cond1 = window(condition_track1, head=seq_head + Config.out_seq_len, lookback=Config.in_seq_len)
                cond2 = window(condition_track2, head=seq_head + 1, lookback=Config.in_seq_len)
                # cond3 = window(condition_track3, head=seq_head + Config.out_seq_len, lookback=Config.in_seq_len)
                y = window(target_track, head=seq_head + 1, lookback=Config.out_seq_len)

                X.append(x)
                Cond.append(cond2)
                Y.append(y)

                # X.append(x), X.append(x), X.append(x)
                # Cond.append(cond1), Cond.append(cond2), Cond.append(cond3)
                # Y.append(y), Y.append(y), Y.append(y)

    X = torch.cat(X)
    Y = torch.cat(Y)
    Cond = torch.cat(Cond)
    return DrumDataset(
        X=X,
        Cond=Cond,
        Y=Y
    )


if __name__ == "__main__":
    dataset = get_dataset()
    dataset.save(Config.dataset_dir)
