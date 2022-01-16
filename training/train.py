import os

import numpy.random
import torchaudio
import torch
import torch.nn as nn

from core.util import format_input_output
from model.LSTM import LSTMModel
from training.audio_utils import to_single_channel, prepare_offset_windows

if __name__ == '__main__':
    print(f'Running training on {torch.cuda.get_device_name(torch.cuda.current_device())}')
    num_epochs = 200

    in_seq_len = 44100
    out_seq_len = 44100

    in_channels = 2
    batch_size = 40
    hidden_dim = 20
    num_layers = 5

    model = LSTMModel(in_channels, hidden_dim, num_layers, out_seq_len, pool_kernel_size=100).cuda()
    total_num_sequences = 0

    error = nn.MSELoss()

    learning_rate = 5e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    input_directory = '../data/split'

    songs = [format_input_output(directory) for directory in os.walk(input_directory)]
    raw_inputs = []
    raw_expected_outputs = []
    inputs = torch.Tensor()
    expected_outputs = torch.Tensor()

    print("Creating batches...")
    for song in songs:
        if 'input' in song.keys():
            print(f"Song: {song['input']}")
            input_no_drums, sample_rate = torchaudio.load(song['input'], normalize=True)
            expected_outputs, _ = torchaudio.load(song['output'], normalize=True)

            input_no_drums = to_single_channel(input_no_drums)
            expected_outputs = to_single_channel(expected_outputs)

            length = input_no_drums.size()[0]  # number of samples in the whole song

            num_sequences = length // out_seq_len
            total_num_sequences += num_sequences

            for seq_number in range(num_sequences):
                seq_head = seq_number * out_seq_len

                combined_input = prepare_offset_windows(input_no_drums, expected_outputs, seq_head, in_seq_len, out_seq_len)
                expected_output = expected_outputs[seq_head:seq_head + out_seq_len].reshape(1, -1)

                raw_inputs.append(combined_input)
                raw_expected_outputs.append(expected_output)

    inputs = torch.cat(raw_inputs, dim=0)
    expected_outputs = torch.cat(raw_expected_outputs, dim=0)

    batch_count = total_num_sequences // batch_size
    print(f"Created {batch_count} batches.\nBeginning training")
    for epoch in range(num_epochs):
        epoch_losses = []
        perm = numpy.random.permutation(inputs.size()[0])
        inputs = inputs[perm]
        expected_outputs = expected_outputs[perm]
        for batch_num in range(batch_count):
            input = inputs[batch_num * batch_size: (batch_num + 1) * batch_size].cuda()
            expected_output = expected_outputs[batch_num * batch_size: (batch_num + 1) * batch_size].cuda()

            optimizer.zero_grad()

            output = model(input)

            loss = error(output, expected_output)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            print(f'Epoch: {epoch} Batch: {batch_num}/{batch_count} Loss: {loss.item()}')
        checkpoint_name = f'epoch{epoch}_inSeqLen{in_seq_len}_outSeqLen{out_seq_len}_batchSize{batch_size}_hiddenDim{hidden_dim}_numLayers{num_layers}'
        with open('losses.txt', 'a') as f:
            f.write(f'Model: {checkpoint_name} - LOSS: {torch.mean(torch.Tensor(epoch_losses))}\n')
        torch.save(model.state_dict(), f'..\\saved_models\\{checkpoint_name}')