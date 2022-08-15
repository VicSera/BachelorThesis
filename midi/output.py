from mido import MidiFile, MidiTrack, Message, MetaMessage, bpm2tempo
from torch.distributions import Categorical

from core.util import denormalize
from midi.util import idx_to_note_or_control


def tensor_to_track(ts):
    track = MidiTrack()
    track.append(MetaMessage('set_tempo', tempo=bpm2tempo(12), time=0))
    track.append(MetaMessage('time_signature', numerator=4, denominator=4, clocks_per_click=24, notated_32nd_notes_per_beat=8, time=0))
    track.append(MetaMessage('key_signature', key='C', time=0))

    for entry in ts:
        data = extract_note_or_control(entry)
        value_or_velocity = denormalize(entry[-2], max=127).int().item() % 128
        time = denormalize(entry[-1], max=4295).int().item()
        if time < 0:
            time = -time
        if data['type'] == 'note_on':
            msg = Message('note_on', channel=9, note=data['value'], velocity=value_or_velocity, time=time)
        else:
            msg = Message('control_change', channel=9, control=data['value'], value=value_or_velocity, time=time)
        track.append(msg)
        print(msg)

    return track


def extract_note_or_control(entry):
    logits = entry[:26]
    distribution = Categorical(logits=logits)
    idx = distribution.sample()
    data = idx_to_note_or_control(idx)
    return data


def tensor_to_midi(ts, filename):
    mid = MidiFile()
    track = tensor_to_track(ts)
    mid.tracks.append(track)

    mid.save(filename)

