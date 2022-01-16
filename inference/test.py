import torch
import torchaudio

from model.LSTM import LSTMModel
from training.audio_utils import prepare_offset_windows, to_single_channel

if __name__ == '__main__':
    epoch = 39

    in_seq_len = 44100
    out_seq_len = 44100
    input_dim = 2

    batch_size = 40
    hidden_dim = 10
    num_layers = 3

    model = LSTMModel(input_dim, hidden_dim, num_layers, out_seq_len, pool_kernel_size=100).cuda()
    model.load_state_dict(torch.load(f'..\\saved_models\\epoch{epoch}_inSeqLen{in_seq_len}_outSeqLen{out_seq_len}_batchSize{batch_size}_hiddenDim{hidden_dim}_numLayers{num_layers}'))
    model.eval()

    waveform, sample_rate = torchaudio.load('..\\data\\split\\01-Explosia\\no-drums.flac')
    waveform = to_single_channel(waveform)

    length = waveform.size()[0]
    num_sequences = length // out_seq_len

    output = torch.Tensor()

    with torch.no_grad():
        for seq_number in range(num_sequences):
            print(f"Processing sequence {seq_number}/{num_sequences}")
            seq_head = seq_number * out_seq_len

            input = prepare_offset_windows(waveform, output, seq_head, in_seq_len, out_seq_len).cuda()

            next_generated_sequence = model(input).flatten().detach()
            next_generated_sequence = next_generated_sequence.to(torch.device("cpu"))
            output = torch.cat([output, next_generated_sequence])

            # if (seq_number % 20 == 0):
            #     torchaudio.save(f'..\\generated_{seq_number}seconds.flac', output.reshape(1, -1), sample_rate, format='flac')

    torchaudio.save('..\\generated.flac', output.reshape(1, -1), sample_rate, format='flac')
