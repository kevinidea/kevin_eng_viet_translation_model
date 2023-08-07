#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1e
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    """
    Class that maps x_char_emb (batch_size, char_embed_size, max_word_length)
    to x_conv_out (batch_size, word_embed_size)
    """
    def __init__(self, max_word_length: int, char_embed_size: int,
               num_filters: int, kernel_size: int = 5):
        """
        character CNN module
        @param max_word_length (int): maximum number of chars in a word
        @param char_embed_size (int): embedding size of each character n a word
        @param n_filters (int): number of output channels produced by the convolution
        @param kernel_size (int): size of the convolving kernel
        """
        super(CNN, self).__init__()
        self.max_word_length = max_word_length
        self.char_embed_size = char_embed_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.conv1d = nn.Conv1d(
            in_channels=self.char_embed_size,
            out_channels=self.num_filters,
            kernel_size=self.kernel_size,
            bias=True
        )
        # torch.max can simply be used in case of global pooling
        # self.max_pool1d = nn.MaxPool1d(kernel_size=max_word_length - self.kernel_size + 1)

    def forward(self, x_char_emb: torch.Tensor) -> torch.Tensor:
        """
        @param x_char_emb (Tensor): padded char embedding with shape
            (batch_size, char_embed_size, max_word_length)
        @return x_conv_out (Tensor): convolution output with shape (batch_size, word_embed_size)
        """
        x_conv_out = F.relu(self.conv1d(x_char_emb))
        x_conv_out = torch.max(x_conv_out, dim=2).values
        # x_conv_out = self.max_pool1d(x_conv_out).squeeze(dim=2)

        return x_conv_out

### END YOUR CODE

