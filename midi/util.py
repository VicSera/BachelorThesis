import torch
from mido import MidiFile

from core.config import Config

FILENAME_INDEX = 7

meta = {
    'max_time': 0  # 4295
}


def note_or_control_to_one_hot(value):
    one_hot = torch.zeros((Config.UNIQUE_NOTES_COUNT + Config.UNIQUE_CONTROLS_COUNT - 1), dtype=torch.int16)

    if value in Config.UNIQUE_NOTES:
        one_hot[Config.UNIQUE_NOTES.index(value)] = 1
    elif value in Config.UNIQUE_CONTROLS:
        idx = Config.UNIQUE_CONTROLS.index(value)
        if idx != len(Config.UNIQUE_CONTROLS) - 1:
            one_hot[idx + Config.UNIQUE_NOTES_COUNT] = 1

    return one_hot


def midi_message_to_tensor(msg):
    if msg.is_cc():
        note_or_control = msg.control
        velocity_or_value = msg.value
    else:
        note_or_control = msg.note
        velocity_or_value = msg.velocity
    time = msg.time

    return torch.tensor([note_or_control, velocity_or_value, time])


def extract_max_time(msg):
    if hasattr(msg, 'time') and msg.time > meta['max_time']:
        meta['max_time'] = msg.time


def extract_unique_notes_and_controls(msg):
    unique_notes, unique_controls = [], []
    if hasattr(msg, 'note') and msg.note not in unique_notes:
        unique_notes.append(msg.note)
    if hasattr(msg, 'control') and msg.control not in unique_controls:
        unique_controls.append(msg.control)


def process_midi_file(file_name):
    print(file_name)
    mid = MidiFile(file_name)
    sequence = []
    sequences = torch.zeros(0, Config.SEQUENCE_LENGTH, 3)

    for i, track in enumerate(mid.tracks):
        for msg in track:
            if msg.type in ['control_change', 'note_on']:
                sequence.append(midi_message_to_tensor(msg))

                if len(sequence) == Config.SEQUENCE_LENGTH:
                    sequence = torch.unsqueeze(torch.stack(sequence), dim=0)
                    sequences = torch.cat((sequences, sequence))
                    sequence = []

    return sequences


def compile_dataset_to_tensor(dataset_root):
    ts = torch.zeros(0, Config.SEQUENCE_LENGTH, 3)

    with open(f'{dataset_root}/info.csv') as info_file:
        info_file.readline()  # skip header
        line = info_file.readline()
        while line:
            info = line.split(',')
            file_path = f'{dataset_root}/{info[Config.FILENAME_INDEX]}'

            res = process_midi_file(file_path)
            ts = torch.cat((ts, res))

            line = info_file.readline()

    return ts


def compile_dataset_to_file(dataset_root):
    ts = compile_dataset_to_tensor(dataset_root)

    torch.save(ts, f'{dataset_root}/dataset.pt')


if __name__ == '__main__':
    compile_dataset_to_file(Config.DATASET_ROOT)
