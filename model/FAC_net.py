import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from models.FF.AutoCorrelation import AutoCorrelationLayer

class FF(nn.Module):
    def __init__(self, configs):
        super(FF, self).__init__()

        # get parameters
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.factor = configs.factor
        self.conv = nn.Conv1d(self.enc_in, self.d_model, kernel_size=1, stride=1)
        self.freq = AutoCorrelationLayer(configs.d_model,configs.n_heads,configs.factor)
        self.avgpool = nn.AvgPool1d(kernel_size=3, stride=1, padding=1)
        input_channels = configs.d_model * configs.seq_len
        self.mlp = nn.Sequential(
                nn.Linear(input_channels, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, 1)
            )
        self.linear = nn.Linear(self.d_model, 1)
    def forward(self, x):
        batch_size = x.shape[0]

        # 1D convolution aggregation
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        new_x = 0
        if not self.configs.data_path.startswith("svmd1"):
            new_x, attn = self.freq(x)
        x = x + new_x
        x = x[:,-1:]
        y = self.linear(x)
        # x = x.reshape(batch_size, -1)
        # y = self.mlp(x)
        # y = y.reshape(batch_size, 1, 1)
        return y