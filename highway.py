#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1d
import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    """
    Class that maps x_conv_out (batch_size, word_embed_size)
    to x_highway (batch_size, word_embed_size)
    """
    def __init__(self, word_embed_size):
        """
        @param word_embed_size (int): embedding size of a word
        """
        super(Highway, self).__init__()
        self.word_embed_size = word_embed_size
        # self.dropout_prob = dropout_prob
        self.proj = nn.Linear(self.word_embed_size, self.word_embed_size, bias=True)
        # xavier vs. kaiming initialization https://pouannes.github.io/blog/initialization/
        #nn.init.xavier_uniform_(self.proj.weight, gain=1)
        nn.init.kaiming_uniform_(self.proj.weight)
        self.gate = nn.Linear(self.word_embed_size, self.word_embed_size, bias=True)
        #nn.init.xavier_uniform_(self.gate.weight, gain=1)
        nn.init.kaiming_uniform_(self.gate.weight)
        # self.dropout = nn.Dropout(p=self.dropout_prob, inplace=False)

    def forward(self, x_conv_out: torch.Tensor) -> torch.Tensor:
        """
        @param x_conv_out (Tensor): a mini batch of convolution output with shape (batch_size, word_embed_size)
        @return: x_highway (Tensor): output with shape (batch_size, word_embed_size)
        """
        x_proj = self.proj(x_conv_out)
        # F.relu gives same output as torch.relu_
        torch.relu_(x_proj)
        # x_proj = F.relu(x_proj)
        x_gate = self.gate(x_conv_out)
        # F.sigmoid gives same output as torch.sigmoid_
        torch.sigmoid_(x_gate)
        # x_gate = F.sigmoid(x_gate)
        x_highway = x_gate * x_proj + (1 - x_gate) * x_conv_out
        # x_word_emb = self.dropout(x_word_emb)

        return x_highway

### END YOUR CODE 

