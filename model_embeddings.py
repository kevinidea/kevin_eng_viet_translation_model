#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
from vocab import VocabEntry

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(f)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module):
    """
    Class that transform words (sents_var (Tensor)): output from to_input_tensor_char from vocab.py where
    each integer is an index into the character vocabulary
    sents_var has shape (max_sentence_length, batch_size, max_word_length)
    into word_embeddings: Tensor of shape (max_sentence_length, batch_size, embed_size),
    containing the CNN-based embeddings for words in a batch of many sentences
    """

    def __init__(self, embed_size: int, vocab: VocabEntry,
                 char_embed_size: int = 50, max_word_length: int = 21,
                 dropout_rate: float = 0.3) -> torch.Tensor:
        """
        Embedding layer for one language
        @param embed_size (int): Embedding size for the output
        @param vocab (VocabEntry): VocabEntry object from vocab.py
        @param char_embed_size (int): Embedding size for the character
        @param max_word_length (int): max number of characters in a word
        @param dropout_rate (float): dropout rate for dropout layer
        """
        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size
        self.vocab = vocab
        self.char_embed_size = char_embed_size
        self.max_word_length = max_word_length
        self.dropout_rate = dropout_rate

        self.char_embedding = nn.Embedding(
            num_embeddings=len(vocab.char2id),
            embedding_dim=self.char_embed_size,
            padding_idx=vocab.char2id['<pad>']
        )

        self.cnn = CNN(
            max_word_length=self.max_word_length,
            char_embed_size=self.char_embed_size,
            num_filters=self.embed_size,
            kernel_size=5
        )

        self.highway = Highway(word_embed_size=embed_size)
        self.dropout = nn.Dropout(p=self.dropout_rate)

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1f

        ### END YOUR CODE

    def forward(self, sents_var: torch.Tensor) -> torch.Tensor:
        """
        Generate word embeddings for words in a batch of sentences
        @param sents_var (Tensor): output from to_input_tensor_char from vocab.py where
            each integer is an index into the character vocabulary
            sents_var has shape (max_sentence_length, batch_size, max_word_length)
        @return word_embeddings (Tensor): Tensor of shape
            (max_sentence_length, batch_size, embed_size)
            containing the CNN-based embeddings for words in a batch of sentences
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        # sents_var has shape of (max_sentence_length, batch_size, max_word_length)
        char_embeddings = self.char_embedding(sents_var)
        # char_embedding return shape of
        # (max_sentence_length, batch_size, max_word_length, char_embed_size)
        # reshape it to
        # (max_sentence_length * batch_size, char_embed_size, max_word_length) to work with cnn layer
        max_sentence_length, batch_size, max_word_length, char_embed_size = char_embeddings.shape
        reshaped_char_embeddings = char_embeddings.view(
            size=(max_sentence_length * batch_size, max_word_length, char_embed_size)
        )
        cnn_char_embeddings = reshaped_char_embeddings.permute(0, 2, 1)

        # x_conv_out has shape (max_sentence_length * batch_size, word_embed_size)
        x_conv_out = self.cnn(cnn_char_embeddings)
        # x_highway has shape (max_sentence_length * batch_size, word_embed_size)
        x_highway = self.highway(x_conv_out)
        # word_embeddings has shape (max_sentence_length * batch_size, word_embed_size)
        word_embeddings = self.dropout(x_highway)
        # reshape word_embeddings to (max_sentence_length, batch_size, word_embed_size)
        word_embeddings = word_embeddings.view(max_sentence_length, batch_size, self.embed_size)

        return word_embeddings

        ### YOUR CODE HERE for part 1f

        ### END YOUR CODE
