import torch
import torchaudio
from torchaudio.transforms import Resample, MelSpectrogram
from tqdm import tqdm

from core.config import Config
from model.LSTM import LSTMModel
from model.modules.upsampling import ConvInUpsampleNetwork
from model.util import sample_from_discretized_mix_logistic
from model.wavenet import WaveNet
from training.audio_utils import prepare_offset_windows, to_single_channel

if __name__ == '__main__':
    epoch = 0

    in_seq_len = 1024
    out_seq_len = 1024
    input_dim = 2

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

    upsample_net = ConvInUpsampleNetwork(
        upsample_scales=Config.upsample_scales,
        cin_channels=Config.num_mels
    )
    model = WaveNet(
        out_channels=Config.out_channels,
        layers=Config.layers,
        stacks=Config.stacks,
        residual_channels=Config.residual_channels,
        gate_channels=Config.gate_channels,
        skip_channels=Config.skip_channels,
        local_conditioning_channels=Config.num_mels,
        dropout_probability=Config.dropout_probability,
        kernel_size=Config.kernel_size,
        upsample_net=upsample_net
    )
    model.load_state_dict(torch.load(f'..\\saved_models\\wavenet_epoch6_batchNum40_withUpsampling_withMelSpec'))
    model.eval()
    model.make_generation_fast_()

    waveform, sample_rate = torchaudio.load('..\\data\\split\\ex2\\no-drums.wav')

    resampler = Resample(sample_rate, Config.sample_rate)
    cond_track = mel_spec(to_single_channel(resampler(waveform))).unsqueeze(0)

    # length = cond_track.size(-1)
    length = 10
    # num_sequences = length // out_seq_len

    output = torch.Tensor()

    with torch.no_grad():
        sequence = model.incremental_forward(c=cond_track[:,:,:length], T=length * Config.hop_size, tqdm=tqdm, softmax=True, quantize=True, log_scale_min=Config.log_scale_min)
        torchaudio.save('..\\generated.flac', sequence.view(1, length * Config.hop_size), Config.sample_rate, format='flac')

    #     for seq_number in range(num_sequences):
    #         print(f"Processing sequence {seq_number}/{num_sequences}")
    #         seq_head = seq_number * out_seq_len
    #
    #         input = prepare_offset_windows(waveform, output, seq_head, in_seq_len, out_seq_len).reshape((2, 1, -1))
    #
    #         next_generated_sequence = model(input[1].reshape(1, 1, -1), input[0].reshape(1, 1, -1))\
    #             .detach()
    #         samples = sample_from_discretized_mix_logistic(next_generated_sequence, log_scale_min=-7.0).flatten()
    #         output = torch.cat([output, samples])
    #
    #         if seq_number % 100 == 0:
    #             torchaudio.save(f'..\\generated_{seq_number}seqs-new.flac', output.reshape(1, -1), sample_rate, format='flac')
    #
    # torchaudio.save('..\\generated.flac', output.reshape(1, -1), sample_rate, format='flac')
