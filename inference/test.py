from core.config import Config
from midi.output import generate_midi_from_scratch
from model.LSTMidi import load_model


if __name__ == '__main__':
    model = load_model(is_gpu=False)

    target_length = 100

    out = generate_midi_from_scratch(model, target_length)
    out.write(f'out_temp{Config.TEMPERATURE}.mid')
