import torch
import torch.nn as nn


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs):
        super(PositionWiseFFN, self).__init__()
        self.layer1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.layer2(self.relu(self.layer1(X)))


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, use_bias=False):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, X, Y):
        if self.use_bias:
            return self.ln(X + Y + self.bias)
        else:
            return self.ln(X + self.dropout(Y))