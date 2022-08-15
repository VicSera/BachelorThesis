import torch
import torch.nn as nn

from core.config import Config


class LSTMidi(nn.Module):
    def __init__(self,
                 one_hot_dim=Config.NOTES_AND_CONTROLS_COUNT,
                 extra_params_dim=Config.EXTRA_PARAMS_COUNT,
                 hidden_dim=Config.HIDDEN_DIM,
                 dropout=Config.DROPOUT,
                 num_layers=Config.NUM_LAYERS):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.combined_dim = one_hot_dim + extra_params_dim

        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(input_size=self.combined_dim,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)

        # Note/control output
        self.out_one_hot = nn.Linear(hidden_dim, one_hot_dim)

        # Extra params output (velocity, value, time)
        self.out_extra_params = nn.Linear(hidden_dim, extra_params_dim)

    def forward(self, x):
        x = x.requires_grad_()
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).cuda()

        _, (hn, _) = self.lstm(x, (h0, c0))

        # Note/control
        out_one_hot = self.out_one_hot(hn[-1])

        # Extra params output
        out_extra_params = self.out_extra_params(hn[-1])

        return out_one_hot, out_extra_params

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        nn.init.uniform(self.embed.weight.data, a=-0.5, b=0.5)
        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)
