# from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import pickle

from collections import OrderedDict
import sys
import time

import numpy as np

import math
import time


import data_process

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


lrate = 0.001
hidden_size = 100
dim_proj = 50
MAX_LENGTH = 20
max_epochs = 30
batch_size = 512
# 1 is added to vocab_size for SOS_token
# vocab_size = 37485 # For Yoochoose
vocab_size = 43098 # For Diginetica
SOS_token = vocab_size-1
EOS_token = 0
dataset='rsc2015'

datasets = {'rsc2015': (data_process.load_data, data_process.prepare_data)}

def get_dataset(name):
    return datasets[name][0], datasets[name][1]

load_data, prepare_data = get_dataset(dataset)
train, valid, test = load_data()
num_train_examples = len(train[0])

def get_pairs_from_dataset(data):
    pairs=[]
    feat_list=data[0]
    lab_list=data[1]
    for feat,lab in zip(feat_list,lab_list):
        pairs.append((feat,lab))
    return pairs

# For Training examples
pairs_train = get_pairs_from_dataset(train)

pairs_test = get_pairs_from_dataset(test)





class EncoderRNN(nn.Module):
    def __init__(self, input_size, dim_proj, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.dim_proj = dim_proj

        self.embedding = nn.Embedding(input_size, dim_proj)
        self.gru = nn.GRU(dim_proj, hidden_size)

    def forward(self, input1, hidden):
        embedded = self.embedding(input1).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)




class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.5, max_length=MAX_LENGTH):
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


def tensor_from_list(indexes):
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = pair[0]
    target_tensor = [pair[1]]
    input_tensor.append(EOS_token)
    target_tensor.append(EOS_token)
    return (tensor_from_list(input_tensor), tensor_from_list(target_tensor))
    



def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)


    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    

    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1) # Topv is the highest softmax value, topi is the corresponding item predicted

        decoder_input = topi.squeeze().detach()  # detach from history as input

        loss += criterion(decoder_output, target_tensor[di])
        if decoder_input.item() == EOS_token:
            break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length 



def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))



def trainIters(encoder, decoder, n_iters, print_every=64, learning_rate=lrate):
    start = time.time()
    # plot_losses = []
    print_loss_total = 0  # Reset every print_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    training_pairs = [tensorsFromPair(i) for i in pairs_train]
    # criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()

    curr_iter = 0

    for iter_out in range(num_train_examples//n_iters):
        
        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[curr_iter+iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            # print("Input Tensor:")
            # print(input_tensor)
            # exit()

            loss = train(input_tensor, target_tensor, encoder,
                         decoder, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))
        curr_iter+=n_iters

    


encoder1 = EncoderRNN(vocab_size, dim_proj, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, vocab_size, dropout_p=0.25).to(device)


# Trains for required number of epochs
for _ in range(max_epochs):
    trainIters(encoder1, attn_decoder1, n_iters = batch_size, print_every=64)    



def evaluate(encoder, decoder, input_tensor, max_length=1):
    with torch.no_grad():
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
        decoder_attentions[di] = decoder_attention.data
        topv, predicted_items = decoder_output.data.topk(20)


        return predicted_items


def compute_score(encoder, decoder):
    mrr=0.0
    recall=0.0
    evalutation_point_count = 0
    for i in pairs_test:
        testing_pair = tensorsFromPair(i)
        input_tensor = testing_pair[0]
        target_tensor = testing_pair[1]
        predicted_items = evaluate(encoder, decoder, input_tensor)
        label = target_tensor[0]
        index_location = (predicted_items == label).nonzero()
        if index_location.shape != torch.Size([0]):
            recall += 1
            mrr+=1/index_location[0]
        evalutation_point_count+=1
    recall /= evalutation_point_count
    mrr /= evalutation_point_count

    return (recall, mrr)
    



recall, mrr = compute_score(encoder1, attn_decoder1)

print("Recall = ",recall)
print("MRR = ",mrr)