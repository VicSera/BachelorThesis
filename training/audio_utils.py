import torch


def to_single_channel(audio_tensor):
    return torch.sum(audio_tensor, 0) / audio_tensor.size()[0]


def prepare_offset_windows(no_drums_track, drum_track, head, length, offset):
    return torch.stack(
        (
            torch.cat((torch.zeros(max(0, length - head - offset)), no_drums_track[max(0, head - length + offset) : head + offset]), dim=0),
            torch.cat((torch.zeros(max(0, length - head)), drum_track[max(0, head - length) : head]), dim=0) if drum_track is not None else torch.zeros(offset)
        ),
        dim=1
    ).reshape(1, -1, 2)