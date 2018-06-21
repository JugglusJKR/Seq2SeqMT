# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division

import config
from data_preprocess import prepareData
from rnn_encoder import EncoderRNN
from attention_decoder import AttnDecoderRNN
from trainer import trainIters
from eval import evaluateRandomly, evaluate, evaluateAndShowAttention

# import for plot
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sys

device = config.get_device()

if __name__ == '__main__':
    # initialization
    hidden_size = 256
    n_layers = 1

    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    print('data preprocessing ok!')
    # initialize encoder and decoder size
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size,
                          n_layers).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words,
                                   dropout_p=0.1).to(device)
    print('model initialization ok!')
    # start training
    trainIters(encoder1, attn_decoder1, 50000,
               input_lang, output_lang, pairs,
               print_every=5000)
    print('training ok!')
    evaluateRandomly(encoder1, attn_decoder1, pairs,
                     input_lang, output_lang)
    print('randomly evaluate ok!')
    output_words, attentions = evaluate(
        encoder1, attn_decoder1,
        input_lang, output_lang,
        "je suis trop froid .")
    plt.matshow(attentions.numpy())
    plt.show()
    print('single evaluate ok!')
    evaluateAndShowAttention(encoder1, attn_decoder1,
                            input_lang, output_lang,
                            "elle a cinq ans de moins que moi .")
    evaluateAndShowAttention(encoder1, attn_decoder1,
                            input_lang, output_lang,
                            "elle est trop petit .")
    evaluateAndShowAttention(encoder1, attn_decoder1,
                            input_lang, output_lang,
                            "je ne crains pas de mourir .")
    evaluateAndShowAttention(encoder1, attn_decoder1,
                            input_lang, output_lang,
                            "c est un jeune directeur plein de talent .")
