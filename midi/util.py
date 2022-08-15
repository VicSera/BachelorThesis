import torch
from mido import MidiFile

from core.config import Config

FILENAME_INDEX = 7

meta = {
    'max_time': 0,  # 4295
    'max_len': None,
    'min_len': None
}


def note_or_control_to_one_hot(value):
    one_hot = torch.zeros(Config.NOTES_AND_CONTROLS_COUNT, dtype=torch.int16)

    if value in Config.UNIQUE_NOTES_AND_CONTROLS:
        one_hot[Config.UNIQUE_NOTES_AND_CONTROLS.index(value)] = 1
    else:
        raise Exception("Invalid value given to note_or_control_to_one_hot: " + value)

    return one_hot


def idx_to_note_or_control(idx):
    if idx < Config.UNIQUE_NOTES_COUNT:
        return {'type': 'note_on', 'value': Config.UNIQUE_NOTES_AND_CONTROLS[idx]}
    else:
        return {'type': 'control_change', 'value': Config.UNIQUE_NOTES_AND_CONTROLS[idx]}


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


def print_midi_file(file_name):
    print(file_name)
    mid = MidiFile(file_name)

    for i, track in enumerate(mid.tracks):
        for msg in track:
            print(msg)

if __name__ == '__main__':
    # print_midi_file('../inference/test.mid')
    # print_midi_file(Config.DATASET_ROOT + '/drummer1/session1/1_funk_80_beat_4-4.mid')
    compile_dataset_to_file(Config.DATASET_ROOT)
    # print(meta)
