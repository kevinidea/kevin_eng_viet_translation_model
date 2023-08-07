#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import re

def pad_sents_char(sents, char_pad_token):
    """ Pad list of sentences according to the longest sentence in the batch and max_word_length.
    @param sents (list[list[list[int]]]): list of sentences, result of `words2charindices()` 
        from `vocab.py`
    @param char_pad_token (int): index of the character-padding token
    @returns sents_padded (list[list[list[int]]]): list of sentences where sentences/words shorter
        than the max length sentence/word are padded out with the appropriate pad token, such that
        each sentence in the batch now has same number of words and each word has an equal 
        number of characters
        Output shape: (batch_size, max_sentence_length, max_word_length)
    """
    # Words longer than 21 characters should be truncated
    max_word_length = 21

    ### YOUR CODE HERE for part 1b
    ### TODO:
    ###     Perform necessary padding to the sentences in the batch similar to the pad_sents() 
    ###     method below using the padding character from the arguments. You should ensure all 
    ###     sentences have the same number of words and each word has the same number of 
    ###     characters. 
    ###     Set padding words to a `max_word_length` sized vector of padding characters.  
    ###
    ###     You should NOT use the method `pad_sents()` below because of the way it handles 
    ###     padding and unknown words.

    max_sent_length = max(len(sent) for sent in sents)
    sents_padded = []
    for sent in sents:
        sent_padded = []
        # each sentence is limited by max_sent_length (not needed but good to show for easy to read)
        for word in sent[:max_sent_length]:
            # each word is limited by max_word_length, truncate longer words
            word_padded = word[:max_word_length]
            # pad shorter words
            word_padded = word_padded + [char_pad_token] * (max_word_length - len(word_padded))
            sent_padded.append(word_padded)
        # pad each sentence so all sentences have equal length if needed
        words_to_add = [[char_pad_token] * max_word_length] * (max_sent_length - len(sent))
        sent_padded.extend(words_to_add)
        sents_padded.append(sent_padded)

    ### END YOUR CODE

    return sents_padded


def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[int]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (int): padding token
    @returns sents_padded (list[list[int]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
        Output shape: (batch_size, max_sentence_length)
    """
    sents_padded = []

    max_len = max(len(s) for s in sents)
    batch_size = len(sents)

    for s in sents:
        padded = [pad_token] * max_len
        padded[:len(s)] = s
        sents_padded.append(padded)

    return sents_padded



def read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents


def separate_sentences(text: str) -> str:
    """
    Separate each sentence with a newline, only separate when there is an actual new sentence
    regex explanation
    (?<![A-Z]) : look behind for no CAPITAL letters,
        this takes care of edge cases like U.S. where it should not be separated
    ([.?!]+) : main one or more delimiter . ? !
    (?!\W*\n) : look ahead for non-word followed by newline
        this prevents multiple newlines being added
    \1 \n : replace with the same delimiter(s) followed by newline
    @param text: str
    @return: str
    """
    return re.sub(r"(?<![A-Z])([.?!]+)(?!\W*\n)", r"\1\n", text)


def clean_sentences(text: str, type: str = 'general_sentences') -> str:
    """
    Generic cleaning for sentences
    @param text: str
    @return: str
    """
    if type == 'general_sentences':
        # Replace multiple line breaks with one line break
        clean_text = re.sub("\n+", "\n", text.strip())
        clean_text = re.sub("\n+\s+\n+", "\n", clean_text)
        # prevent double space
        clean_text = re.sub(r"[ ]+", " ", clean_text)
    elif type == 'output_sentences':
        # Replace all &apos; with ' to be consistent with training data
        clean_text = re.sub("&apos;", "'", text.strip())
        # remove any space in front of punctuations
        clean_text = re.sub("\s+(?=[!#$%\'()*+,-./:;<=>?@[\\]^_`{|}~])", "", clean_text)
        # prevent any double space
        clean_text = re.sub(r"[ ]+", " ", clean_text.strip())
        # remove the wrong "Xin lỗi" output translation for empty space
        clean_text = re.sub(r"^Xin lỗi$", '', clean_text)

    return clean_text

def read_input_sentences(text: str) -> List[List[str]]:
    """
    Take separated sentences by \n, clean them, and transform them into a list of list of words
    with <s> and </s> in front and back per sentence respectively
    @param text: str
    @return: List[List[str]]
    """
    data = []
    # Split the sentences by newline
    sentences = re.split("\n", text)
    for sentence in sentences:
        # Replace all ' with &apos; to be consistent with training data
        sentence = re.sub("'", ' &apos;', sentence)
        # Add a white space before punctuation
        sentence = re.sub('([!\"#$%\'()*+,-./:<=>?@[\\]^_`{|}~])', r' \1 ', sentence)
        # add <s> and </s> to front and back of the sentence
        sentence = "<s> " + sentence + " </s>"
        # prevent double space
        sentence = re.sub(r'[ ]+', ' ', sentence)
        # Turn sentence into list of words
        words = sentence.split(' ')
        data.append(words)
    return data


if __name__ == '__main__':
    # add a unit test to debug pad_sents_char
    sents = [
        [[1, 5, 5, 5, 5, 2], [1, 5, 5, 2]],
        [[1, 5, 2], [1, 5, 5, 5, 5, 5, 5, 5, 2], [1, 5, 5, 5, 5, 2], [1, 5, 2]]
    ]
    print(pad_sents_char(sents, 0))

    sents = """Based on current natural language processing "standard", \
    Kevin's system has done an amazing English Vietnamese translation!
    This is another sentence."""

    print(read_input_sentences(sents))

    sents = """Based on current natural language processing standard, \
    Kevin  's system has done an amazing English Vietnamese translation   !
    This is another sentence, which  is not clean, and with grammar error   ;    """

    print(clean_sentences(sents, 'output_sentences'))