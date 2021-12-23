import torch
import torchaudio

from model.LSTM import LSTMModel

if __name__ == '__main__':
    epoch = 3
    seq_len = 500000

    input_dim = 1
    hidden_dim = 10
    layer_dim = 1
    output_dim = 2

    model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim).cuda()
    model.load_state_dict(torch.load(f'..\\saved_models\\epoch{epoch}_seqSize{seq_len}'))
    model.eval()

    waveform, sample_rate = torchaudio.load('..\\data\\split\\01-Explosia\\no-drums.flac')

    length = waveform.size()[1]
    num_sequences = length // seq_len

    output = torch.Tensor()

    for seq_number in range(num_sequences):
        seq_start = seq_number * seq_len
        seq_end = min((seq_number + 1) * seq_len, length)
        trimmed_waveform = waveform[:, seq_start:seq_end].cuda().reshape((-1, 3, 1))

        trimmed_output = model(trimmed_waveform).reshape((2, -1)).detach()
        trimmed_output = trimmed_output.to(torch.device("cpu"))
        output = torch.cat([output, trimmed_output], dim=1)

    torchaudio.save('..\\generated.flac', output, sample_rate, format='flac')
