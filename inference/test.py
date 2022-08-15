import torch

from core.config import Config
from midi.output import tensor_to_track, tensor_to_midi
from model.LSTMidi import LSTMidi

if __name__ == '__main__':
    epoch = 199
    batch_num = 300
    model_name = "LSTMidi"

    model = LSTMidi()
    model.load_state_dict(torch.load(f'..\\saved_models\\{model_name}_sesh2\\{model_name}_Epoch{epoch}_BatchNum{batch_num}'))
    model.eval()
    model = model.cuda()

    length = 100
    pred = torch.zeros(1, 1, Config.NOTES_AND_CONTROLS_COUNT + Config.EXTRA_PARAMS_COUNT)
    pred = pred.cuda()

    with torch.no_grad():
        while pred.size(1) < length:
            next_node = torch.cat(model(pred), dim=1).unsqueeze(0)
            pred = torch.cat((pred, next_node), dim=1)

    pred = pred[0, 1:]
    tensor_to_midi(pred, 'test.mid')
