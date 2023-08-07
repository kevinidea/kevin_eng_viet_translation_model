#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###     Hint:
        ###     - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###     - Set the padding_idx argument of the embedding matrix.
        ###     - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.

        # note that this is length of the characters set, not words set because we are predicting char
        super(CharDecoder, self).__init__()
        vocab_char_size = len(target_vocab.char2id)
        self.target_vocab = target_vocab
        self.decoderCharEmb = nn.Embedding(
            num_embeddings=vocab_char_size,
            embedding_dim=char_embedding_size,
            padding_idx=target_vocab.char2id['<pad>']
        )
        # decoder can only be forward direction
        self.charDecoder = nn.LSTM(
            input_size=char_embedding_size, hidden_size=hidden_size, bias=True, bidirectional=False
        )
        self.char_output_projection = nn.Linear(
            in_features=hidden_size, out_features=vocab_char_size, bias=True
        )
        ### END YOUR CODE

    def forward(self, input, dec_hidden_before=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden_before: internal state of the LSTM before reading the input characters.
            A tuple of two tensors of shape (1, batch, hidden_size)

        @return scores: called s in the PDF, shape (length, batch, self.vocab_size)
        @return dec_hidden_after: internal state of the LSTM after reading the input characters.
            A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.

        # feed input that has shape (length, batch) to decoderCharEmb
        # return char_embeddings with shape (length, batch, char_embedding_size)
        char_embeddings = self.decoderCharEmb(input)
        # feed char_embeddings (length, batch, char_embedding_size) into LSTM layer
        # return hiddens (length, batch, hidden_size),
        # Tuple(last_hidden, last_cell) each tensor has shape (1, batch, hidden_size)
        # note that (hiddens[-1]==last_hidden).all() should be True
        hiddens, (last_hidden, last_cell) = self.charDecoder(input=char_embeddings, hx=dec_hidden_before)
        # feed hiddens (length, batch, hidden_size) to linear layer
        # return scores (length, batch, vocab_char_size)
        scores = self.char_output_projection(hiddens)
        dec_hidden_after = (last_hidden, last_cell)

        return scores, dec_hidden_after
        
        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden_before=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch).
            Note that "length" here and in forward() need not be the same.
        @param dec_hidden_before: initial internal state of the LSTM, obtained from the output of the word-level decoder.
            A tuple of two tensors of shape (1, batch, hidden_size)

        @return cross_entropy_loss, tensor of a scalar, computed as the *sum* of cross-entropy losses
            of all the words in the batch, for every character in the sequence.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###     Hint:
        ###     - Make sure padding characters do not contribute to the cross-entropy loss.
        ###     - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).

        # run forward pass with char_sequence (length - 1, batch) without <END>
        # return scores (length - 1, batch, vocab_char_size) and dec_hidden_after (1, batch, hidden_size)
        scores, dec_hidden_after = self.forward(char_sequence[:-1], dec_hidden_before)
        # calculate CrossEntropyLoss (ignoring padding characters and sum loss, not average loss)
        loss_func = nn.CrossEntropyLoss(
            ignore_index=self.target_vocab.char2id['<pad>'], reduction='sum'
        )
        # reshape scores with 3 dimensions to have 2 dimensions of shape ( (length - 1) * batch, vocab_char_size)
        # scores contain individual scores for all characters in a vocab_char_size in last dimension
        reshaped_scores = scores.view(-1, scores.shape[-1])
        # create target from char_sequence[1:] without <START>
        # reshape 2 dimensions of shape (length - 1, batch) to 1 dimension of ((length -1) * batch)
        # target contains only index of the right characters
        target = char_sequence[1:].contiguous().view(-1)
        cross_entropy_loss = loss_func(reshaped_scores, target)

        return cross_entropy_loss

        ### END YOUR CODE

    def decode_greedy(self, initial_states, device, max_length=21):
        """ Greedy decoding
        @param initial_states: initial internal state of the LSTM,
            a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @return decoded_words: a list (of length batch) of strings, each of which has length <= max_length.
            The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ###     Hints:
        ###     - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###     - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###     - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###       Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.

        start_idx = self.target_vocab.start_of_word
        end_idx = self.target_vocab.end_of_word
        dec_hidden = initial_states  # Tuple of two tensors of size (1, batch, hidden_size)
        batch_size = dec_hidden[0].shape[1]
        # create batch of start chars with shape (length=1, batch)
        current_char_batch = torch.tensor([[start_idx] * batch_size], device=device)
        output_words, decoded_words = [], []

        for t in range(max_length):
            # dec_hidden variable has to be the same for recursively looping through it
            # forward returns scores of shape (length=1, batch, vocab_char_size)
            scores, dec_hidden = self.forward(current_char_batch, dec_hidden)
            # current_char_batch has shape (length=1, batch) since argmax pool at the last dimension
            current_char_batch = scores.argmax(dim=-1)
            output_words.append(current_char_batch)
        # output_words is a list[Tensor with shape (1, batch)] with length of max_length

        # truncate output_words by removing ending char }
        # turn output_words to list with len of batch and element as list with length of max_length
        # output_words_ls has shape of (batch, max_length) but saved as a list for looping
        output_words_ls = torch.cat(output_words).permute(1, 0).tolist()
        for batch in output_words_ls:
            word = ''
            # generate 1 word per batch, each word either ends at end_idx of } or at max_length
            for char in batch:
                if char == end_idx:
                    break
                word += self.target_vocab.id2char[char]
            # decoded words should contain batch_size number of words
            decoded_words.append(word)

        return decoded_words
        ### END YOUR CODE

