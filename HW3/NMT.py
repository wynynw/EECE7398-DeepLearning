from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import os
from collections import Counter
import argparse
import time
import math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
import numpy as np


class Lang:
    def __init__(self, name, vocab_path):
        self.name = name
        self.word2index = {'<unk>': 2}
        self.index2word = {0: "SOS", 1: "EOS", 2: '<unk>'}
        self.n_words = 3  # Count <unk>, SOS and EOS
        self.vocab_path = vocab_path

    def loadVocab(self):
        with open(self.vocab_path, encoding="utf-8") as f:
            rawData = f.readlines()
            vocab = list(map(lambda word: word[:-1], rawData))
        vocab_norm = [normalizeString(word) for word in vocab]
        unique_vocab_norm = list(Counter(vocab_norm).keys())
        for word in unique_vocab_norm:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


# Turn a Unicode string to plain ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def readLangs(lang1, lang2, vocab1=None, vocab2=None, reverse=False):
    print("Reading lines...")
    # Read the file and split into lines
    lines1 = open(lang1, encoding='utf-8').read().strip().split('\n')
    lines2 = open(lang2, encoding='utf-8').read().strip().split('\n')

    lines = [line1 + '\t' + line2 for line1, line2 in zip(lines1, lines2)]

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2, vocab2)
        output_lang = Lang(lang1, vocab1)
    else:
        input_lang = Lang(lang1, vocab1)
        output_lang = Lang(lang2, vocab2)

    return input_lang, output_lang, pairs


def filterPair(p, MAX_LENGTH=50):
    return len(p[0].split(' ')) < MAX_LENGTH and \
           len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs, max_length=50):
    return [pair for pair in pairs if filterPair(pair, max_length)]


def prepareData(lang1, lang2, vocab1=None, vocab2=None, reverse=False, max_length=50):
    print('loading data from: ' + lang1)
    print('loading data from: ' + lang2)
    input_lang, output_lang, pairs = readLangs(lang1, lang2, vocab1=vocab1, vocab2=vocab2, reverse=reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs, max_length)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    input_lang.loadVocab()
    output_lang.loadVocab()
    print("Counted words:")
    print("In vocabulary " + input_lang.vocab_path + ': ', input_lang.n_words)
    print("In vocabulary " + output_lang.vocab_path + ': ', output_lang.n_words)
    return input_lang, output_lang, pairs


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=50):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def indexesFromSentence(lang, sentence):
    indexes = []
    for word in sentence.split(' '):
        if word in lang.word2index:
            indexes.append(lang.word2index[word])
        else:
            indexes.append(lang.word2index['<unk>'])
    return indexes


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(decoder.max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    if not os.path.isdir('model'):
        os.mkdir('model')

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()
    old_loss=0
    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        if iter > 1:
            if old_loss > loss:
                torch.save(encoder.state_dict(), './model/encoder' + model_name + '.pt')
                torch.save(decoder.state_dict(), './model/decoder' + model_name + '.pt')

        print_loss_total += loss
        plot_loss_total += loss

        if iter == 1 or iter == n_iters + 1:
            print('Loss '+str(iter)+':'+str(loss))
            np.save(model_name + '_train_loss', plot_losses)

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('Loss '+str(iter)+':'+str(print_loss_avg))
        old_loss = loss
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


def evaluate(encoder, decoder, sentence):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        max_length = decoder.max_length
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=5):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def testBlueScore(encoder, decoder, pairs):
    smooth = SmoothingFunction().method1
    blue = []
    for pair in pairs:
        try:
            output_words, attentions = evaluate(encoder, decoder, pair[0])
        except RuntimeError:
            pass
        output_sentence = ' '.join(output_words)
        blue.append(sentence_bleu([output_sentence], pair[1], smoothing_function=smooth))
    blue_mean = np.mean(blue)
    return blue_mean


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Neural Machine Translation')
    parser.add_argument('mode', type=str, help='Mode: train / test / translate')
    #parser.add_argument('--lang', type=str, default='V', help='{C: Czech, G: German,V: Vietnamese}')
    parser.add_argument('--gpu', type=str, default='0', help='GPU index from 0, 1, 2')

    arg = parser.parse_args()

    device = torch.device("cuda:" + arg.gpu if torch.cuda.is_available() else "cpu")

    SOS_token = 0
    EOS_token = 1

    MAX_LENGTH = 50

    teacher_forcing_ratio = 0.8

    model_name = 'Vitnamese'
    TRAIN_PATH = ['./dataset/English-Vietnamese/trainset/trainen.txt',
                  './dataset/English-Vietnamese/trainset/trainvi.txt']
    VOCAB_PATH = ['dataset/English-Vietnamese/vocabularies/vocaben.txt',
                  'dataset/English-Vietnamese/vocabularies/vocabvi.txt']
    TEST_PATH = ['./dataset/English-Vietnamese/testset/tst2013en.txt',
                 './dataset/English-Vietnamese/testset/tst2013vi.txt']

    input_lang, output_lang, pairs = prepareData(TRAIN_PATH[0], TRAIN_PATH[1], VOCAB_PATH[0], VOCAB_PATH[1], False,
                                                 max_length=MAX_LENGTH)

    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.2).to(device)
    if arg.mode == 'train':
        trainIters(encoder1, attn_decoder1, 50000, print_every=5000)
        evaluateRandomly(encoder1, attn_decoder1)

    elif arg.mode == 'test':
        encoder1.load_state_dict(torch.load('model/encoder' + model_name + '.pt'))
        attn_decoder1.load_state_dict(torch.load('model/decoder' + model_name + '.pt'))
        _, _, test_pair = prepareData(TEST_PATH[0], TEST_PATH[1], VOCAB_PATH[0], VOCAB_PATH[1], False,
                                      max_length=MAX_LENGTH)
        print("BLEU score:" + str(testBlueScore(encoder1, attn_decoder1, test_pair)))
    elif arg.mode == 'translate':
        print("Loading Model....")
        encoder1.load_state_dict(torch.load('model/encoder' + model_name + '.pt'))
        attn_decoder1.load_state_dict(torch.load('model/decoder' + model_name + '.pt'))
        while True:
            print(">",end=' ')
            string = str(input())
            decoded_words, _ = evaluate(encoder1, attn_decoder1, string)
            output_sentence = ' '.join(decoded_words)
            print(output_sentence)
