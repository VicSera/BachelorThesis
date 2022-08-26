import os
from random import randint

import pretty_midi
import torch
import torch.nn.functional as F
from pretty_midi import Note

from core.config import Config
from core.util import normalize


def note_to_tensor(note: Note, prev_start):
    step = note.start - prev_start
    duration = note.end - note.start
    velocity = normalize(note.velocity, Config.MAX_VELOCITY)
    return torch.tensor((note.pitch, velocity, step, duration), dtype=torch.float32)


def normalize_note_tensor(ts):
    return torch.div(ts, torch.tensor([Config.NOTES_COUNT, 1, 1, 1]))


def sequence_to_input(sequence):
    if len(sequence.shape) == 3:  # Batched input
        pitch = F.one_hot(sequence[:, :, 0].long(), num_classes=Config.NOTES_COUNT).float()
        normalized = normalize_note_tensor(sequence)
    else:  # Single input
        pitch = F.one_hot(sequence[:, 0].long(), num_classes=Config.NOTES_COUNT).float()
        normalized = normalize_note_tensor(sequence)

    return pitch, normalized


def get_sequence(midi, start_idx=None):
    notes = sorted(midi.instruments[0].notes, key=lambda n: n.start)
    if start_idx is None:
        start_idx = randint(0, len(notes) - Config.SEQUENCE_LENGTH)
    prev_start = notes[start_idx - 1].start if start_idx > 0 else 0

    tensors = []

    for note in notes[start_idx:start_idx + Config.SEQUENCE_LENGTH]:
        note_ts = note_to_tensor(note, prev_start)
        prev_start = note.start
        tensors.append(note_ts)

    return normalize_note_tensor(torch.stack(tensors)).unsqueeze(0)


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

        if len(current_sequence) == Config.SEQUENCE_LENGTH:
            current_ts = torch.stack(current_sequence).unsqueeze(0)
            sequences.append(current_ts)
            current_sequence = current_sequence[Config.SEQUENCE_HOP:]

    return sequences


def compile_dataset(dataset_root, csv_path):
    sequences = {split: [] for split in Config.SPLITS}

    line_num = 0
    with open(csv_path, 'r', errors='replace') as info_file:
        info_file.readline() # skip header
        line_num += 1
        line = info_file.readline()
        while line:
            info = line.split(',')
            file_path = f'{dataset_root}/{info[Config.FILENAME_INDEX]}'
            split = info[Config.SPLIT_INDEX]
            print(line_num, file_path)

            res = compile_midi_to_sequences(file_path)
            sequences[split] += res

            line = info_file.readline()
            line_num += 1

    return {split: torch.cat(sequence) for split, sequence in sequences.items()}


if __name__ == '__main__':
    if not os.path.exists(Config.DATASET_DIR):
        os.makedirs(Config.DATASET_DIR)

    dataset = compile_dataset(Config.DATASET_ROOT, Config.CSV_PATH)
    for split, data in dataset.items():
        torch.save(data, Config.DATASET[split])
