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

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)

        # Note/control output
        self.note_out = nn.Linear(hidden_dim, num_notes)

        # Extra params output (velocity, value, time)
        self.extra_params_out = nn.Linear(hidden_dim, num_extra_params)

    def forward(self, x):
        x = x.requires_grad_()
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        if next(self.parameters()).is_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()

        _, (hn, _) = self.lstm(x, (h0, c0))

        # Note/control
        note = self.note_out(hn[-1])

        # Extra params output
        extra_params = self.extra_params_out(hn[-1])

        return note, extra_params


def load_model(is_eval=True, is_gpu=True):
    model = LSTMidi()
    model.load_state_dict(torch.load(f'{Config.MODEL_DIR}\\{Config.MODEL_NAME}_{Config.SESSION}\\{Config.CHECKPOINT}'))
    if is_eval:
        model.eval()
    if is_gpu:
        return model.cuda()
    return model
