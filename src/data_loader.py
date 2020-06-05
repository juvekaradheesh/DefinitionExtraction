import os
import random
import pandas as pd
import torch
from torchtext import data

# FULL
# def_data_dir = os.path.join('..', 'data', 'def_data', 'full')
# train_data_path = os.path.join(def_data_dir, 'train')
# valid_data_path = os.path.join(def_data_dir, 'val')

# WCL
# def_data_dir = os.path.join('..', 'data', 'wcl', 'full')
# train_data_path = os.path.join(def_data_dir, 'train')
# valid_data_path = os.path.join(def_data_dir, 'val')

# W00
def_data_dir = os.path.join('..', 'data', 'w00', 'full')
train_data_path = os.path.join(def_data_dir, 'train')
valid_data_path = os.path.join(def_data_dir, 'val')

SEED = 2020
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def file_to_list(path):
    list_ = []
    f = open(path, 'r')
    lines = f.readlines()
    for line in lines:
        list_.append(line.replace('\n',''))
    f.close()
    
    return list_

def get_data(data_path):
    path = os.path.join(data_path, 'sentences.txt')
    sentences = file_to_list(path)
    
    path = os.path.join(data_path, 'labels.txt')
    labels = file_to_list(path)
    
    return sentences, labels

def load_data(train_data_path=train_data_path, valid_data_path=valid_data_path, is_csv=False, batch_size=64):
    if not is_csv:
        # Reading data from file
        print('Reading data from file')
        train_data = get_data(train_data_path)
        valid_data = get_data(valid_data_path)
        print('Training data sentences: ', len(train_data[0]))
        print('Validation data sentences: ', len(valid_data[0]))

        # Put data to dataframe and then save to a csv file
        df_train = pd.DataFrame(list(zip(train_data[0], train_data[1])), 
                        columns =['Sentences', 'Labels'])
        df_valid = pd.DataFrame(list(zip(valid_data[0], valid_data[1])), 
                        columns =['Sentences', 'Labels'])

        # df = pd.concat([df_train, df_val])
        train_data_path = os.path.join(train_data_path, 'def_data_train.csv')
        valid_data_path = os.path.join(valid_data_path, 'def_data_val.csv')
        df_train.to_csv(train_data_path)
        df_valid.to_csv(valid_data_path)

        # Get Data statistics
        non_def = df_train.groupby('Labels').count()['Sentences'][0]
        def_ = df_train.groupby('Labels').count()['Sentences'][1]
        print("Definition, non-definition sentences in training data: ", def_, ", ", non_def)

        non_def = df_valid.groupby('Labels').count()['Sentences'][0]
        def_ = df_valid.groupby('Labels').count()['Sentences'][1]
        print("Definition, non-definition sentences in validation data: ", def_, ", ", non_def)

    TEXT = data.Field(tokenize='spacy', batch_first=True, include_lengths=True)
    LABEL = data.LabelField(dtype=torch.float, batch_first=True)

    fields = [(None, None), ('sentence',TEXT),('label', LABEL)]

    train_data = data.TabularDataset(path=train_data_path, format='csv', fields=fields, skip_header = True)
    valid_data = data.TabularDataset(path=valid_data_path, format='csv', fields=fields, skip_header = True)

    print("Example from training data: \n", vars(train_data.examples[0]))

    # train_data, valid_data = training_data.split(split_ratio=0.7, random_state = random.seed(SEED))

    TEXT.build_vocab(train_data,min_freq=3,vectors = "glove.6B.100d")
    LABEL.build_vocab(train_data)
    # print("Size of TEXT vocabulary:",len(TEXT.vocab))
    # print("Size of LABEL vocabulary:",len(LABEL.vocab))
    # print(TEXT.vocab.freqs.most_common(10))  
    # print(TEXT.vocab.stoi)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

    train_iterator, valid_iterator = data.BucketIterator.splits(
        (train_data, valid_data), 
        batch_size = batch_size,
        sort_key = lambda x: len(x.sentence),
        sort_within_batch=True,
        device = device)

    return train_iterator, valid_iterator, TEXT, LABEL

if __name__ == "__main__":
    load_data()