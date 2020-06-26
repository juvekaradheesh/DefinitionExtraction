import os
import random
import string

import pandas as pd
import numpy as np
import torch
from torchtext import data
from torch.utils.data import Dataset, DataLoader
from gensim.models.keyedvectors import KeyedVectors

SEED = 2020
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class TaggingDataset():

    def __init__(self, data_path):
        self.word2idx = {}
        self.lbl2idx = {'B':0, 'I':1, 'O':2, 'E':3, 'S':4, '-1':5}
        self.char2idx = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’’’/\|_@#$%ˆ&*̃‘+-=<>()[]{}'

        self.sentences = self.file_to_list(os.path.join(data_path, 'sentences.txt'))
        self.labels = self.file_to_list(os.path.join(data_path, 'labels.txt'))
        
        print('Getting GloVe Embeddings')
        self.glove_matrix = self.getGloveEmbeddings('utils/glove.6B.100d.txt')
        
        self.max_sen = 0
        self.max_word = 0
        self.samples = []

        self.init_dataset()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def getGloveEmbeddings(self, embeddings_path):
        vectors = []
        
        with open(embeddings_path, 'r') as f:
            for idx, l in enumerate(f):
                line = l.split()
                word = line[0]
                line = line[1:]
                self.word2idx[word] = idx
                vectors.append([float(i) for i in line])

        vectors = np.array(vectors)

        # Randomly initialized vector for OOV
        vectors = np.vstack((vectors, np.random.rand(100)))
        # 0 vector for padding
        vectors = np.vstack((vectors, np.zeros(100)))
        # Convert to tensor
        vectors = torch.from_numpy(vectors)

        return vectors

    def sentence_to_list(self):
        for i, sentence in enumerate(self.sentences):
            self.sentences[i] = sentence.split(' ')
            
    def labels_to_list(self):
        for i, label in enumerate(self.labels):
            self.labels[i] = label.split(' ')

    def get_max_sent_word_len(self):
        for s in self.sentences:
            if len(s) > self.max_sen:
                self.max_sen = len(s)
            for w in s:
                if len(w) > self.max_word:
                    self.max_word = len(w)
    
    def pad_sent_labels(self):
        for i, (sentence, label) in enumerate(zip(self.sentences, self.labels)):
            if len(sentence) < self.max_sen:
                diff = self.max_sen - len(sentence)
                padding = []
                label_padding = []
                for j in range(diff):
                    padding.append('<pad>')
                    label_padding.append('-1')
                temp_sent = sentence + padding
                temp_label = label + label_padding
                self.sentences[i] = temp_sent
                self.labels[i] = temp_label

    def init_char(self, word):
        sequence = []
        if word == '<pad>':
            sequence = [96]*self.max_word
            return sequence
        for c in word:
            idx = self.char2idx.find(c)
            if idx == -1:
                # OOV
                sequence.append(96)
            else:
                sequence.append(idx)
        diff_w = self.max_word - len(word)
        for i in range(diff_w):
            # Padding
            sequence.append(97)
        return sequence

    def init_dataset(self):
        print('Initializing dataset')
        print('\tSentences to list')
        self.sentence_to_list()
        self.labels_to_list()
        self.get_max_sent_word_len()
        print('\tPad sentences')
        self.pad_sent_labels()
        print('\tList to idx')
        for sentence, label in zip(self.sentences, self.labels):
            feed_sentence = []
            feed_labels = []
            feed_char = []
            for word, label in zip(sentence,label):
                feed_labels.append(self.lbl2idx[label])
                if word == '<pad>':
                    # Padders
                    feed_sentence.append(400001)
                elif word in self.word2idx:
                    vocab_obj = self.word2idx[word]
                    feed_sentence.append(vocab_obj)
                else:
                    # Out-of-Vocabulary (OOV)
                    feed_sentence.append(400000)
                feed_char.append(self.init_char(word))
            self.samples.append((torch.tensor(feed_sentence), torch.tensor(feed_char), torch.tensor(feed_labels)))

    def file_to_list(self, path):
        list_ = []
        f = open(path, 'r')
        lines = f.readlines()
        for line in lines:
            list_.append(line.replace('\n',''))
        f.close()
        
        return list_

if __name__ == "__main__":
    def_data_dir = os.path.join('data', 'tagging', 'openstax')
    train_data_path = os.path.join(def_data_dir, 'train')
    valid_data_path = os.path.join(def_data_dir, 'val')

    dataset = TaggingDataset(train_data_path)
    print(len(dataset))
    print(dataset[420])

    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    for batch in iter(train_loader):
        print(batch)
        print(batch[0].size(), batch[1].size(), batch[2].size())
        print(batch[0][0], batch[1][0], batch[2][0])
        break