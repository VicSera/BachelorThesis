import pretty_midi

from core.config import Config
from midi.dataset import MidiDataset
from midi.output import generate_midi_from_scratch, generate_midi
from midi.util import get_sequence
from model.LSTMidi import load_model


if __name__ == '__main__':
    Config.load_from_file(Config.EVAL_INFO_FILE)
    model = load_model(is_gpu=False)

    dataset = MidiDataset.load(Config.DATASET['train'])

    target_length = 100

    if Config.USE_EXAMPLE:
        midi = pretty_midi.PrettyMIDI(Config.EXAMPLE_FILE)
        start_sequence = get_sequence(midi, 0)
        out = generate_midi(model, start_sequence, target_length)
    else:
        out = generate_midi_from_scratch(model, target_length)
    out.write(f'{Config.WEIGHT_FILE}-{"ex-train" if Config.USE_EXAMPLE else "noex"}-temp{Config.TEMPERATURE}.mid')
