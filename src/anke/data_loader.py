import os
import string
import numpy as np


import torch
from torch.utils.data import Dataset, DataLoader
from gensim.models.keyedvectors import KeyedVectors

SEED = 2020
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class ClassificationDataset(Dataset):

    def __init__(self, data_path, embedding_type, embeddings_path):
        self.embeddings_path = embeddings_path
        self.embedding_type = embedding_type
        self.sentences = self.file_to_list(os.path.join(data_path, 'sentences.txt'))
        self.labels = self.file_to_list(os.path.join(data_path, 'labels.txt'))
        if embedding_type == 'w2v':
            self.w2v = KeyedVectors.load_word2vec_format(embeddings_path, binary=True)
        else:
            self.word2idx = {}
            self.glove_matrix = self.getGloveEmbeddings()

        self.max_sen = 0
        self.samples = []

        # Initialization of dataset
        self.remove_punctuations()
        self.init_dataset()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    def getGloveEmbeddings(self):
        vectors = []
        
        with open(self.embeddings_path, 'r') as f:
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

    def remove_punctuations(self):
        for i, s in enumerate(self.sentences):
            s = s.translate(str.maketrans('', '', string.punctuation))
            sent_list = s.split(' ')
            while '' in sent_list:
                sent_list.remove('')
            self.sentences[i] = sent_list

    def get_max_sent_len(self):
        for s in self.sentences:
            if len(s) > self.max_sen:
                self.max_sen = len(s)

    def pad_sentences(self):
        for i, s in enumerate(self.sentences):
            if len(s) < self.max_sen:
                diff = self.max_sen - len(s)
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
                if word == '<pad>':
                    # Padders
                    feed_sentence.append(3000001 if self.embedding_type=='w2v' else 400001)
                elif word in (self.w2v.vocab if self.embedding_type=='w2v' else self.word2idx):
                    vocab_obj = (self.w2v.vocab[word] if self.embedding_type=='w2v' else self.word2idx[word])
                    feed_sentence.append(vocab_obj.index if self.embedding_type=='w2v' else vocab_obj)
                else:
                    # Out-of-Vocabulary (OOV)
                    feed_sentence.append(3000000 if self.embedding_type=='w2v' else 400000)

            self.samples.append((torch.tensor(feed_sentence), int(label)))

    def file_to_list(self, path):
        list_ = []
        f = open(path, 'r')
        lines = f.readlines()
        for line in lines:
            list_.append(line.replace('\n',''))
        f.close()
        
        return list_

if __name__ == "__main__":
    def_data_dir = os.path.join('data', 'classification', 'w00')
    train_data_path = os.path.join(def_data_dir, 'train')
    valid_data_path = os.path.join(def_data_dir, 'val')
    embedding_type = 'glove'
    embeddings_path = 'utils/glove.6B.100d.txt'

    dataset = ClassificationDataset(train_data_path, embedding_type, embeddings_path)
    print(len(dataset))
    print(dataset[420])

    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    for batch in iter(train_loader):
        print(batch)
        print(len(batch[0]), batch[0][0].size(), batch[1].size())
        break