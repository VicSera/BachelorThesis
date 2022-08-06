import torch
import torch.nn as nn

from core.config import Config


class LSTMidi(nn.Module):
    def __init__(self,
                 one_hot_dim=Config.UNIQUE_NOTES_COUNT + Config.UNIQUE_CONTROLS_COUNT - 1,
                 extra_params_dim=Config.EXTRA_PARAMS_COUNT,
                 hidden_dim=Config.HIDDEN_DIM,
                 num_layers=Config.NUM_LAYERS):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.combined_dim = one_hot_dim + extra_params_dim

        # self.combine_layer = nn.Linear(, combined_dim)

        self.lstm = nn.LSTM(input_size=self.combined_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)

        # Note/control output
        self.out_one_hot = nn.Linear(hidden_dim, one_hot_dim)
        self.softmax = nn.Softmax(dim=-1)

        # Extra params output (velocity, value, time)
        self.out_extra_params = nn.Linear(hidden_dim, extra_params_dim)
        self.out_relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().cuda()

        _, (hn, _) = self.lstm(x, (h0.detach(), c0.detach()))
        x = hn[-1]

        # Note/control output
        out_one_hot = self.out_one_hot(x)
        out_one_hot = self.softmax(out_one_hot)

        # Extra params output
        out_extra_params = self.out_extra_params(x)
        out_extra_params = self.out_relu(out_extra_params)

        return out_one_hot, out_extra_params

