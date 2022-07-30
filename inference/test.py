import torch
import torchaudio

from core.config import Config
from model.wavenet_2 import WaveNetModel

if __name__ == '__main__':
    epoch = 6
    batch_num = 80
    model_name = "wavenet2_ed3_withConv"

    model = WaveNetModel()
    model.load_state_dict(torch.load(f'..\\saved_models\\{model_name}\\wavenet_epoch{epoch}_batchNum{batch_num}'))
    model.eval()

    # waveform, sample_rate = torchaudio.load('..\\data\\split\\ex2\\no-drums.wav')

    # resampler = Resample(sample_rate, Config.sample_rate)
    # cond_track = mel_spec(to_single_channel(resampler(waveform))).unsqueeze(0)

    # length = cond_track.size(-1)
    length = 10
    # num_sequences = length // out_seq_len

    output = torch.Tensor()

    with torch.no_grad():
        def prog_callback(step, total_steps):
            print(str(100 * step // total_steps) + "% generated")


        generated = model.generate_fast(num_samples=160000,
                                        progress_callback=prog_callback,
                                        progress_interval=1000,
                                        temperature=1.0,
                                        regularize=0.)
        generated = torch.from_numpy(generated).reshape(1, -1)
        # sequence = model.incremental_forward(c=cond_track[:,:,:length], T=length * Config.hop_size, tqdm=tqdm, softmax=True, quantize=True, log_scale_min=Config.log_scale_min)
        torchaudio.save('..\\generated.flac', generated, Config.sample_rate, format='flac')

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
