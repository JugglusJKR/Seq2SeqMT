# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division

import random
import config

import torch
from data_preprocess import tensorFromSentence
from result_show import showAttention

######################################################################
# Evaluation
# ==========
#
# Evaluation is mostly the same as training, but there are no targets so
# we simply feed the decoder's predictions back to itself for each step.
# Every time it predicts a word we add it to the output string, and if it
# predicts the EOS token we stop there. We also store the decoder's
# attention outputs for display later.
#

MAX_LENGTH = config.get_max_len()
device = config.get_device()
SOS_TOKEN = config.get_sos_token()
EOS_TOKEN = config.get_eos_token()

def evaluate(encoder, decoder,
             input_lang, output_lang,
             sentence,
             max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_TOKEN]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_TOKEN:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


######################################################################
# We can evaluate random sentences from the training set and print out the
# input, target, and output to make some subjective quality judgements:
#

def evaluateRandomly(encoder, decoder, pairs,
                     input_lang, output_lang,
                     n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder,
                                            input_lang, output_lang,
                                            pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def evaluateAndShowAttention(encoder, decoder,
                            input_lang, output_lang,
                            input_sentence):
    output_words, attentions = evaluate(encoder, decoder,
                                        input_lang, output_lang,
                                        input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)

