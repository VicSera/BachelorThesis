import os

import torchaudio
from torch.utils.data import Dataset
import torch
from torchaudio.transforms import Resample, MelSpectrogram

from core.config import Config
from core.util import format_input_output
from training.audio_utils import to_single_channel
from training.util import window


class DrumDataset(Dataset):
    def __init__(self, X, Y, Cond):
        self.X = X
        self.Y = Y
        self.Cond = Cond

    def __getitem__(self, item):
        return self.X[item].unsqueeze(0), self.Cond[item], self.Y[item].unsqueeze(1)

    def __len__(self):
        return len(self.X)


def get_dataset():
    X, Cond, Y = [], [], []

    mel_spec = MelSpectrogram(
        sample_rate=Config.sample_rate,
        n_fft=Config.n_fft,
        win_length=Config.win_length,
        hop_length=Config.hop_size,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        onesided=True,
        n_mels=Config.num_mels
    )

    cond_len = Config.in_seq_len // Config.hop_size
    songs = [format_input_output(directory) for directory in os.walk(Config.input_directory)]
    print("Creating batches...")
    for song in songs:
        if 'input' in song.keys():
            print(f"Song: {song['input']}")
            condition_track, sr = torchaudio.load(song['input'], normalize=True)
            target_track, _ = torchaudio.load(song['output'], normalize=True)

            # Preprocess
            resampler = Resample(sr, Config.sample_rate)
            condition_track = resampler(condition_track)
            target_track = resampler(target_track)

            condition_track = mel_spec(to_single_channel(condition_track))
            target_track = to_single_channel(target_track)

            length = target_track.size()[-1]  # number of samples in the whole song

            num_sequences = length // Config.out_seq_len

            for seq_number in range(num_sequences):
                seq_head = seq_number * Config.out_seq_len
                cond_head = seq_number * cond_len

                x = window(target_track, head=seq_head + Config.out_seq_len, lookback=Config.in_seq_len)
                cond = window(condition_track, head=cond_head, lookback=cond_len, dims=2)
                y = window(target_track, head=seq_head, lookback=Config.out_seq_len)

                X.append(x)
                Cond.append(cond)
                Y.append(y)

    return DrumDataset(
        X=torch.cat(X),
        Cond=torch.cat(Cond),
        Y=torch.cat(Y)
    )
