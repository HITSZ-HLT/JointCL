# -*- coding: utf-8 -*-
# file: data_utils.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.

import os
import pickle
import numpy as np
import random
import torch
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer
from collections import Counter
from copy import deepcopy
from glob import glob
import math
import json
from random import sample
import pandas as pd
import argparse
TOTAL_ASPECTS = ["food","menu","ambience","miscellaneous","place","service","price","staff",
                 "FOOD#QUALITY","RESTAURANT#PRICES","AMBIENCE#GENERAL","RESTAURANT#MISCELLANEOUS",
                 "FOOD#PRICES","DRINKS#QUALITY","DRINKS#PRICES","RESTAURANT#GENERAL","FOOD#STYLE_OPTIONS","SERVICE#GENERAL"]

num2polarity = {0: 'AGAINST', 1: 'FAVOR', 2: 'NONE'}

def build_tokenizer_old(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "

        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer

def split_punc(text):
    processed_text = ""
    for c in text:
        if c in [',','.','!','?','/','#','@','(',')','{','}']:
            processed_text += ' '
        processed_text += c
    return processed_text

def build_vast_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            data_file = pd.read_csv(fname)
            for i in data_file.index:
                row = data_file.iloc[i]
                aspect = row['topic_str']

                text = split_punc(row['text_s'])

                text_raw = text + " " + aspect+ " "
                text += text_raw
        tokenizer = Tokenizer(max_seq_len)
        # print(text)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def build_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines)):
                line = lines[i].strip()
                line_data = line.split("\t")
                assert len(line_data)==2
                # text_raw = line_data[0]
                text_raw = split_punc(line_data[0])
                aspect = fname.split("/")[-1].lower()
                aspect.replace("_"," ")
                # assert aspect in TOTAL_ASPECTS,"{} is new".format(aspect)
                text_raw = text_raw + " " + aspect.replace("#"," ")+ " "
                text += text_raw
        tokenizer = Tokenizer(max_seq_len)
        # print(text)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer


def build_naacl_tokenizer(fnames, max_seq_len, dat_fname):
    if os.path.exists(dat_fname):
        print('loading tokenizer:', dat_fname)
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        text = ''
        for fname in fnames:
            file = pd.read_csv(fname)
            for idx, row in file.iterrows():
                text_raw = row['tweet']
                topic = ' '.join(json.loads(row['topic']))
                if type(text_raw) is not str:
                    continue
                text_raw = text_raw + " " + topic + " "
                text +=  text_raw
        tokenizer = Tokenizer(max_seq_len)
        # print(text)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer

def _load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = './glove.twitter.27B/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else '../glove.42B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x



class Tokenizer(object):
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {'[mask]':1}
        self.idx2word = {1:'[mask]'}
        self.idx = 2

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post',max_seq_len=None):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx)+1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        if max_seq_len is None:
            max_seq_len = self.max_seq_len
        return pad_and_truncate(sequence, max_seq_len, padding=padding, truncating=truncating)


class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name):
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        # self.tokenizer.add_special_tokens()
        self.max_seq_len = max_seq_len

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post',max_seq_len=None):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        if len(sequence) == 0:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        if max_seq_len is None:
            max_seq_len = self.max_seq_len
        return pad_and_truncate(sequence, max_seq_len, padding=padding, truncating=truncating)

class TraditionDataset():
    def __init__(self, data_dir, tasks, tokenizer, opt):
        self.tasks = tasks
        self.all_targets = self.tasks
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.opt = opt
        self.polarities = opt.polarities
        self.get_all_aspect_indices()
        self.all_data = []
        self.all_data = self._get_all_data()

    def __getitem__(self, index):
        return self.all_data[index]

    def __len__(self):
        return len(self.all_data)

    def get_all_aspect_indices(self,):
        self.all_aspect_indices = []
        for aspect in self.all_targets:
            aspect  = aspect.replace("#",' ')
            aspect = aspect.replace("_",' ')
            aspect_indices = self.tokenizer.text_to_sequence(aspect,max_seq_len=4)
            assert not all(aspect_indices==0), "aspect error"
            self.all_aspect_indices.append(aspect_indices)
        assert len(self.all_aspect_indices)==len(self.tasks)


    def _get_all_data(self):
        self.polarity2label = { polarity:idx for idx, polarity in enumerate(self.opt.polarities)}
        self.label2polarity = { idx:polarity for idx, polarity in enumerate(self.opt.polarities)}
        out_masked_features = "./masked_feature/{}".format(self.opt.dataset)
        if not os.path.exists(out_masked_features):
            os.makedirs(out_masked_features)
        index = 0
        for task_id,task in enumerate(self.tasks):
            fname = os.path.join(self.data_dir,task)
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            fin.close()
            aspect = task.split(".")[0].replace("#",' ').lower()
            aspect = aspect.replace("_"," ")
            for i in range(0, len(lines)):
                line  = lines[i].strip()
                line_data = line.split("\t")
                assert len(line_data)==2,"line data error"
                text = split_punc(line_data[0])
                polarity = line_data[1]
                text_indices = self.tokenizer.text_to_sequence(text)
                aspect_indices = self.tokenizer.text_to_sequence(aspect,max_seq_len=4)
                aspect_len = np.sum(aspect_indices != 0)
                if self.polarity2label.get(polarity) is not None:
                    label = self.polarity2label[polarity]
                else:
                    print("error polarity: {}\n{}".format(len(polarity),self.polarity2label))
                    label=None
                assert type(label)==int
                text_len = np.sum(text_indices != 0)
                concat_bert_indices = self.tokenizer.text_to_sequence('[CLS] ' + text + ' [SEP] ' + aspect + " [SEP]")
                concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
                concat_segments_indices = pad_and_truncate(concat_segments_indices, self.tokenizer.max_seq_len)
                text_bert_indices = self.tokenizer.text_to_sequence("[CLS] " + text + " [SEP]")
                data = {
                    'concat_bert_indices': concat_bert_indices,
                    'concat_segments_indices': concat_segments_indices,
                    'text_bert_indices': text_bert_indices,
                    'text_indices': text_indices,
                    'aspect_indices': aspect_indices,
                    'topic_index': task_id,
                    'polarity': label,
                    'index': index,
                }
                index += 1
                self.all_data.append(data)

        return self.all_data


class NaacalDataset():
    def __init__(self, fname, tokenizer, opt):
        self.fname = fname
        self.tokenizer = tokenizer
        self.opt = opt
        self.polarities = opt.polarities
        self.all_data = []
        self.all_data = self._get_all_data()

    def __getitem__(self, index):
        return self.all_data[index]

    def __len__(self):
        return len(self.all_data)

    def _get_all_data(self):
        out_masked_features = "./masked_feature/{}".format(self.opt.dataset)
        if not os.path.exists(out_masked_features):
            os.makedirs(out_masked_features)
        file = pd.read_csv(self.fname)
        topics = list(file['topic'].drop_duplicates())
        topic2index = {topic: idx for idx, topic in enumerate(topics)}
        for idx, row in file.iterrows():
            topic_index = topic2index[row['topic']]
            topic = ' '.join(json.loads(row['topic']))
            text = row['tweet']
            if type(text) is not str:
                continue
            label = row['label']
            assert type(label) == int
            if label not in self.polarities:
                continue
            text_indices = self.tokenizer.text_to_sequence(text)
            target_indices = self.tokenizer.text_to_sequence(topic, max_seq_len=4)
            target_len = np.sum(target_indices != 0)
            text_len = np.sum(text_indices != 0)
            concat_bert_indices = self.tokenizer.text_to_sequence('[CLS] ' + text + ' [SEP] ' + topic + " [SEP]")
            concat_segments_indices = [0] * (text_len + 2) + [1] * (target_len + 1)
            concat_segments_indices = pad_and_truncate(concat_segments_indices, self.tokenizer.max_seq_len)
            text_bert_indices = self.tokenizer.text_to_sequence("[CLS] " + text + " [SEP]")
            data = {
                'concat_bert_indices': concat_bert_indices,
                'concat_segments_indices': concat_segments_indices,
                'text_bert_indices': text_bert_indices,
                'text_indices': text_indices,
                'aspect_indices': target_indices,
                'polarity': label,
                'topic_index': topic_index,
                'index': idx,
            }
            self.all_data.append(data)
        return self.all_data

class ZeroshotDataset():
    def __init__(self, data_dir, tokenizer, opt,data_type ):

        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.opt = opt
        self.polarities = opt.polarities
        # self.get_all_aspect_indices()
        self.all_data = []
        self.data_type = data_type
        self.type = opt.type
        self.all_data = self._get_all_data()


    def __getitem__(self, index):
        return self.all_data[index]

    def __len__(self):
        return len(self.all_data)


    def _get_all_data(self):
        out_masked_features = "./masked_feature/{}".format(self.opt.dataset)
        if not os.path.exists(out_masked_features):
            os.makedirs(out_masked_features)

        data_file = pd.read_csv(self.data_dir)
        topics = list(data_file['topic_str'].drop_duplicates())
        topic2index = { topic:idx for idx, topic in enumerate(topics)}
        all_tasks = list(data_file['topic_str'])
        if self.data_type == 'train' or self.type ==2:

            for i in data_file.index:
                row = data_file.iloc[i]
                aspect = row['topic_str']
                topic_index = topic2index[aspect]

                text = split_punc(row['text_s'])
                label = int(row['label'])

                aug_aspect = random.sample(all_tasks, 1)[0]
                aug_aspect_indices = self.tokenizer.text_to_sequence(aug_aspect, max_seq_len=4)

                text_indices = self.tokenizer.text_to_sequence(text)
                aspect_indices = self.tokenizer.text_to_sequence(aspect, max_seq_len=4)
                aspect_len = np.sum(aspect_indices != 0)
                aug_aspect_len = np.sum(aug_aspect_indices != 0)

                text_len = np.sum(text_indices != 0)
                concat_bert_indices = self.tokenizer.text_to_sequence('[CLS] ' + text + ' [SEP] ' + aspect + " [SEP]")
                concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
                concat_segments_indices = pad_and_truncate(concat_segments_indices, self.tokenizer.max_seq_len)
                text_bert_indices = self.tokenizer.text_to_sequence("[CLS] " + text + " [SEP]")

                aug_concat_bert_indices = self.tokenizer.text_to_sequence(
                    '[CLS] ' + text + ' [SEP] ' + aug_aspect + " [SEP]")
                aug_concat_segments_indices = [0] * (text_len + 2) + [1] * (aug_aspect_len + 1)
                aug_concat_segments_indices = pad_and_truncate(aug_concat_segments_indices, self.tokenizer.max_seq_len)

                data = {
                    'concat_bert_indices': concat_bert_indices,
                    'concat_segments_indices': concat_segments_indices,
                    'aug_concat_bert_indices': aug_concat_bert_indices,
                    'aug_concat_segments_indices': aug_concat_segments_indices,
                    'text_bert_indices': text_bert_indices,
                    'text_indices': text_indices,
                    'aspect_indices': aspect_indices,
                    'polarity': label,
                    'topic_index':topic_index,
                    'index':i,
                }
                self.all_data.append(data)

        else:
            for i in data_file.index:
                row = data_file.iloc[i]
                if int(row['seen?']) == self.type:
                    aspect = row['topic_str']

                    text = split_punc(row['text_s'])
                    topic_index = topic2index[aspect]
                    label = int(row['label'])
                    aug_aspect = random.sample(all_tasks, 1)[0]
                    aug_aspect_indices = self.tokenizer.text_to_sequence(aug_aspect, max_seq_len=4)

                    text_indices = self.tokenizer.text_to_sequence(text)
                    aspect_indices = self.tokenizer.text_to_sequence(aspect, max_seq_len=4)
                    aspect_len = np.sum(aspect_indices != 0)
                    aug_aspect_len = np.sum(aug_aspect_indices != 0)

                    text_len = np.sum(text_indices != 0)
                    concat_bert_indices = self.tokenizer.text_to_sequence(
                        '[CLS] ' + text + ' [SEP] ' + aspect + " [SEP]")
                    concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
                    concat_segments_indices = pad_and_truncate(concat_segments_indices, self.tokenizer.max_seq_len)
                    text_bert_indices = self.tokenizer.text_to_sequence("[CLS] " + text + " [SEP]")

                    aug_concat_bert_indices = self.tokenizer.text_to_sequence(
                        '[CLS] ' + text + ' [SEP] ' + aug_aspect + " [SEP]")
                    aug_concat_segments_indices = [0] * (text_len + 2) + [1] * (aug_aspect_len + 1)
                    aug_concat_segments_indices = pad_and_truncate(aug_concat_segments_indices,
                                                                   self.tokenizer.max_seq_len)

                    data = {
                        'concat_bert_indices': concat_bert_indices,
                        'concat_segments_indices': concat_segments_indices,
                        'aug_concat_bert_indices': aug_concat_bert_indices,
                        'aug_concat_segments_indices': aug_concat_segments_indices,
                        'text_bert_indices': text_bert_indices,
                        'text_indices': text_indices,
                        'aspect_indices': aspect_indices,
                        'polarity': label,
                        'topic_index':topic_index,
                        'index':i,
                    }
                    self.all_data.append(data)

        return self.all_data




def _get_tasks(task_path):
    tasks = []
    with open(task_path) as file:
        for line in file.readlines():
            line = line.strip()
            tasks.append(line)
    return tasks

def _get_file_names(data_dir,tasks):
    return [ os.path.join(data_dir,task) for task in tasks]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

