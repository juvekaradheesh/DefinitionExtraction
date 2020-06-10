import os
import random
import string

import pandas as pd
import torch
from torchtext import data
from torch.utils.data import Dataset, DataLoader
from gensim.models.keyedvectors import KeyedVectors

SEED = 2020
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class ClassificationDataset(Dataset):

    def __init__(self, data_path):
        self.sentences = self.file_to_list(os.path.join(data_path, 'sentences.txt'))
        self.labels = self.file_to_list(os.path.join(data_path, 'labels.txt'))
        self.w2v = KeyedVectors.load_word2vec_format('../utils/GoogleNews-vectors-negative300.bin', binary=True)
        self.max_len = 0
        self.samples = []

        # Initialization of dataset
        self.remove_punctuations()
        self.init_dataset()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    def remove_punctuations(self):
        for i, s in enumerate(self.sentences):
            s = s.translate(str.maketrans('', '', string.punctuation))
            sent_list = s.split(' ')
            while '' in sent_list:
                sent_list.remove('')
            self.sentences[i] = sent_list

    def get_max_sent_len(self):
        for s in self.sentences:
            if len(s) > self.max_len:
                self.max_len = len(s)

    def pad_sentences(self):
        for i, s in enumerate(self.sentences):
            if len(s) < self.max_len:
                diff = self.max_len - len(s)
                padding = []
                for j in range(diff):
                    padding.append('<pad>')
                temp = s + padding
                self.sentences[i] = temp

    def init_dataset(self):
        self.get_max_sent_len()
        self.pad_sentences()
        for sentence, label in zip(self.sentences, self.labels):
            feed_sentence = []
            for word in sentence:
                if word in self.w2v.vocab:
                    vocab_obj = self.w2v.vocab[word]
                    feed_sentence.append(vocab_obj.index)
                else:
                    feed_sentence.append(3000000)
            self.samples.append((feed_sentence, int(label)))

    def file_to_list(self, path):
        list_ = []
        f = open(path, 'r')
        lines = f.readlines()
        for line in lines:
            list_.append(line.replace('\n',''))
        f.close()
        
        return list_

class TaggingDataset():

    def __init__(self):
        pass
    
    def __len__(self):
        pass

    def __getitem__(self):
        pass

if __name__ == "__main__":
    def_data_dir = os.path.join('..', 'data', 'classification', 'w00')
    train_data_path = os.path.join(def_data_dir, 'train')
    valid_data_path = os.path.join(def_data_dir, 'val')

    dataset = ClassificationDataset(train_data_path)
    print(len(dataset))
    print(dataset[420])

    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
