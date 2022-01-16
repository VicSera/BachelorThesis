import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, pool_kernel_size):
        super(LSTMModel, self).__init__()

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.num_layers = num_layers

        self.pool_kernel_size = pool_kernel_size

        # Max Pool
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size)

        # LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True)  # batch_first=True (batch_dim, seq_dim, feature_dim)

        # Linear layer
        self.fc = nn.Linear(hidden_dim, output_dim // pool_kernel_size)


    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().cuda()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().cuda()

        out = self.pool(x.reshape(x.size(0), 2, -1))
        out = out.reshape(x.size(0), -1, 2)

        _, (hn, _) = self.lstm(out, (h0.detach(), c0.detach()))

        out = self.fc(hn[-1])
        out = torch.repeat_interleave(out, self.pool_kernel_size, dim=1)
        return out


