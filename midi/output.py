import pretty_midi
import torch
from torch.distributions import Categorical

from core.config import Config
from core.util import denormalize, clamp


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
    generated = start_sequence
    with torch.no_grad():
        while generated.size(1) < target_length:
            prediction = torch.cat(model(generated), dim=1).unsqueeze(0)
            logits = prediction[:, :, :Config.NOTES_COUNT] / Config.TEMPERATURE
            pitch = Categorical(logits=logits).sample().unsqueeze(0) / Config.NOTES_COUNT
            next_note = torch.cat((pitch, prediction[:, :, Config.NOTES_COUNT:]), dim=2)
            generated = torch.cat((generated, next_note), dim=1)
    generated = generated.squeeze(dim=0)
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