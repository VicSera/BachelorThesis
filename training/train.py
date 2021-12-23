import os
from pathlib import Path

import torchaudio
import torch
import torch.nn as nn

from core.util import format_input_output
from model.LSTM import LSTMModel

if __name__ == '__main__':
    print(f'Running training on {torch.cuda.get_device_name(torch.cuda.current_device())}')
    num_epochs = 20
    prev_msg_len = 0

    seq_len = 500000
    input_dim = 1
    hidden_dim = 10
    layer_dim = 1
    output_dim = 2
    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim).cuda()

    error = nn.MSELoss()

    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    input_directory = '../data/split'

    songs = [format_input_output(directory) for directory in os.walk(input_directory)]
    for epoch in range(num_epochs):
        losses = []
        for idx, song in enumerate(songs):
            if 'input' in song.keys():
                input_waveform, sample_rate = torchaudio.load(song['input'])
                expected_waveform, _ = torchaudio.load(song['output'])

                length = input_waveform.size()[1]
                num_sequences = length // seq_len

                song_losses = []
                for seq_number in range(num_sequences):
                    seq_start = seq_number * seq_len
                    seq_end = min((seq_number + 1) * seq_len, length)
                    trimmed_input_waveform = input_waveform[:,seq_start:seq_end].cuda().reshape((-1, 3, 1))
                    trimmed_expected_waveform = expected_waveform[:,seq_start:seq_end].cuda()

                    optimizer.zero_grad()

                    output = model(trimmed_input_waveform).reshape((2, -1))

                    loss = error(output, trimmed_expected_waveform)
                    message = f'Epoch: {epoch} Song: {idx}/{len(songs)} Sequence: {seq_number}/{num_sequences} Loss: {loss.data.item()}'
                    losses.append(loss.data.item())
                    song_losses.append(loss.data.item())

                    # print('\b' * prev_msg_len + message)

                    prev_msg_len = len(message)
                    loss.backward()

                    optimizer.step()

                print(f"Epoch: {epoch} Song: {idx}/{len(songs)} Loss: {torch.mean(torch.Tensor(song_losses))}")
        print(f'Loss for epoch {epoch} is: {torch.mean(torch.Tensor(losses))}')
        torch.save(model.state_dict(), f'..\\saved_models\\epoch{epoch}_seqSize{seq_len}')