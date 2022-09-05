import torch
import torch.nn as nn

from core.config import Config


class LSTMidi(nn.Module):
    def __init__(self,
                 num_notes=Config.NOTES_COUNT,
                 num_extra_params=Config.EXTRA_PARAMS_COUNT,
                 hidden_dim=Config.HIDDEN_DIM,
                 dropout=Config.DROPOUT,
                 num_layers=Config.NUM_LAYERS):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_size = 1 + num_extra_params

        self.relu = nn.ReLU()

        self.lstm1 = nn.LSTM(input_size=num_notes,
                             hidden_size=hidden_dim,
                             num_layers=num_layers,
                             dropout=dropout,
                             batch_first=True)
        self.lstm2 = nn.LSTM(input_size=self.input_size,
                             hidden_size=hidden_dim,
                             num_layers=num_layers,
                             dropout=dropout,
                             batch_first=True)

        # Note/control output
        self.note_out = nn.Linear(hidden_dim, num_notes)

        # Extra params output (velocity, value, time)
        self.extra_params_out = nn.Linear(hidden_dim, num_extra_params)

    def forward(self, x):
        pitch, extra = x
        hidden1_0 = torch.zeros(self.num_layers, pitch.size(0), self.hidden_dim)
        cell1_0 = torch.zeros(self.num_layers, pitch.size(0), self.hidden_dim)
        hidden2_0 = torch.zeros(self.num_layers, extra.size(0), self.hidden_dim)
        cell2_0 = torch.zeros(self.num_layers, extra.size(0), self.hidden_dim)
        if next(self.parameters()).is_cuda:
            hidden1_0 = hidden1_0.cuda()
            cell1_0 = cell1_0.cuda()
            hidden2_0 = hidden2_0.cuda()
            cell2_0 = cell2_0.cuda()

        _, (hidden1_n, _) = self.lstm1(pitch, (hidden1_0, cell1_0))
        _, (hidden2_n, _) = self.lstm2(extra, (hidden2_0, cell2_0))

        # Note/control
        note = self.note_out(hidden1_n[-1])

        # Extra params output
        extra_params = self.extra_params_out(hidden2_n[-1])

        return note, extra_params


def load_model(file=Config.WEIGHTS_PATH, is_eval=True, is_gpu=True):
    model = LSTMidi(num_notes=Config.NOTES_COUNT,
                    num_extra_params=Config.EXTRA_PARAMS_COUNT,
                    hidden_dim=Config.HIDDEN_DIM,
                    dropout=Config.DROPOUT,
                    num_layers=Config.NUM_LAYERS)
    if is_gpu:
        loaded = torch.load(file)
    else:
        loaded = torch.load(file, map_location=torch.device('cpu'))
    model.load_state_dict(loaded)

    if is_eval:
        model.eval()
    if is_gpu:
        return model.cuda()
    return model
