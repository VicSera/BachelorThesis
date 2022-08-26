import pretty_midi
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from core.config import Config
from core.util import denormalize, clamp
from midi.util import sequence_to_input


def tensor_to_midi(ts):
    mid = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(
        program=pretty_midi.instrument_name_to_program(Config.INSTRUMENT)
    )

    prev_start = 0

    for entry in ts:
        norm_pitch, norm_velocity, step, duration = entry
        start = max(0, (prev_start + step).item())
        end = max(0, (start + duration).item())
        velocity = clamp(denormalize(norm_velocity, Config.MAX_VELOCITY).int().item())
        pitch = clamp(denormalize(norm_pitch, Config.NOTES_COUNT).int().item())
        prev_start = start

        note = pretty_midi.Note(
            pitch=pitch,
            velocity=velocity,
            start=start,
            end=end
        )

        instrument.notes.append(note)

    mid.instruments.append(instrument)

    return mid


def generate_midi(model, start_sequence, target_length):
    pitch, note_data = sequence_to_input(start_sequence)
    generated = note_data
    with torch.no_grad():
        while generated.size(1) < target_length:
            pitch = pitch[:, -Config.SEQUENCE_LENGTH:, :]
            note_data = generated[:, -Config.SEQUENCE_LENGTH:, :]

            next_pitch, next_extra = model((pitch, note_data))

            logits = next_pitch / Config.TEMPERATURE
            next_pitch = Categorical(logits=logits).sample().unsqueeze(0)
            next_pitch_one_hot = F.one_hot(next_pitch, num_classes=Config.NOTES_COUNT)

            pitch = torch.cat((pitch, next_pitch_one_hot), dim=1)
            next_note_data = torch.cat((next_pitch / Config.NOTES_COUNT, next_extra), dim=1)
            generated = torch.cat((generated, next_note_data.unsqueeze(0)), dim=1)
    generated = generated.squeeze(dim=0)[start_sequence.size(1):]

    return tensor_to_midi(generated.cpu())


def generate_midi_from_scratch(model, target_length):
    start_sequence = torch.zeros((1, 1, 4))
    return generate_midi(model, start_sequence, target_length)


def extract_midi_node_dict(mid):
    notes = mid.instruments[0].notes
    return [{'start': note.start,
             'end': note.end,
             'pitch': note.pitch,
             'velocity': note.velocity} for note in notes]
