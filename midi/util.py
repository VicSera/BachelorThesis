import pretty_midi
import torch
from pretty_midi import Note

from core.config import Config
from core.util import normalize


def note_to_tensor(note: Note, prev_start):
    step = note.start - prev_start
    duration = note.end - note.start
    velocity = normalize(note.velocity, Config.MAX_VELOCITY)
    return torch.tensor((note.pitch, velocity, step, duration), dtype=torch.float32)


def compile_midi_to_sequences(filename):
    mid = pretty_midi.PrettyMIDI(filename)
    current_sequence = []
    sequences = []

    notes = sorted(mid.instruments[0].notes, key=lambda n: n.start)
    prev_start = 0
    for note in notes:
        note_ts = note_to_tensor(note, prev_start)
        prev_start = note.start
        current_sequence.append(note_ts)

        if len(current_sequence) == Config.SEQUENCE_LENGTH + Config.SEQUENCE_HOP:
            current_sequence = current_sequence[Config.SEQUENCE_HOP:]
            current_ts = torch.stack(current_sequence).unsqueeze(0)
            sequences.append(current_ts)

    return sequences


def compile_dataset(dataset_root, csv_path):
    sequences = []

    with open(csv_path, 'r', errors='replace') as info_file:
        info_file.readline()  # skip header
        line = info_file.readline()
        while line:
            info = line.split(',')
            file_path = f'{dataset_root}/{info[Config.FILENAME_INDEX]}'
            print(file_path)

            res = compile_midi_to_sequences(file_path)
            sequences += res

            line = info_file.readline()

    return torch.cat(sequences)


if __name__ == '__main__':
    dataset = compile_dataset(Config.DATASET_ROOT, Config.CSV_PATH)
    torch.save(dataset, Config.TRAINING_DATASET)