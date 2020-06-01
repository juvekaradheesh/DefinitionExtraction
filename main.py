import os
import random
import json
from sys import argv
from argparse import ArgumentParser

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from torch.utils.tensorboard import SummaryWriter

from src.models.sentence_classifier import SentenceClassifier
from src.data_loader import load_data
from src.train import train, evaluate

def getopts(argv):
    opts = {}
    while argv:
        if argv[0][0] == '-' and argv[0][1] != 'h':
            opts[argv[0]] = argv[1]
        argv = argv[1:]
    return opts

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('-M', '--mode', help='select mode to run', required=True, choices=['train', 'test'])
    parser.add_argument('-m', '--modelpath', help='provide save/load path for the model', required=True)
    parser.add_argument('-t', '--task', help='select task to run', required=False, choices=['tagging', 'classification'], default='classification')
    parser.add_argument('-a', '--architecture', help='select model architecture to use', required=False, choices=['anke', 'mahovy', 'bert'], default='anke')
    parser.add_argument('-d', '--dataset', help='select dataset to use', required=False, choices=['w00', 'wcl', 'all'], default='wcl')
    parser.add_argument('-p', '--param_file_path', help='provide path to hyperparameter json file', required=False, default='utils/params_classification.json')


    # hyperparameters
    # parser.add_argument('-p', '--param_file_path', help='provide path to hyperparameter json file', required=False, default='utils/params_classification.json')

    myargs = vars(parser.parse_args())
    print(myargs)
    with open(myargs['param_file_path']) as f:
        params = json.load(f)

    # tensorboard summary writer
    writer = SummaryWriter()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

    if myargs['mode'] == 'train':
        # Load data
        print("Loading data")    
        if myargs['dataset'] == 'all':
            def_data_dir = os.path.join('data', myargs['task'], 'all')
            train_data_path = os.path.join(def_data_dir, 'train')
            valid_data_path = os.path.join(def_data_dir, 'val')
            is_csv = False
        
        if myargs['dataset'] == 'wcl':
            def_data_dir = os.path.join('data', myargs['task'], 'wcl')
            train_data_path = os.path.join(def_data_dir, 'train')
            valid_data_path = os.path.join(def_data_dir, 'val')
            is_csv = False

        if myargs['dataset'] == 'w00':
            def_data_dir = os.path.join('data', myargs['task'], 'w00')
            train_data_path = os.path.join(def_data_dir, 'train')
            valid_data_path = os.path.join(def_data_dir, 'val')
            is_csv = False
        

        train_iterator, valid_iterator, TEXT, LABEL = load_data(train_data_path, valid_data_path, is_csv, params['batch_size'])

        # Define parameters
        params['num_embeddings'] = len(TEXT.vocab)

        #Initialize the pretrained embedding
        glove_vectors = TEXT.vocab.vectors

        # Instantiate the model
        if myargs['architecture'] == 'anke':
            model = SentenceClassifier(params, glove_vectors)

        # Model Architecture
        print(model)

        # No. of trianable parameters
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
            
        print(f'The model has {count_parameters(model):,} trainable parameters')

        # Define optimizer and loss
        if params['optimizer'] == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        if params['loss_fn'] == 'bce':
            criterion = nn.BCELoss()

        # Push to cuda if available
        model = model.to(device)
        criterion = criterion.to(device)

        best_valid_loss = float('inf')

        for epoch in range(params['num_epochs']):
            
            # Train the model
            train_loss, train_acc, train_f1 = train(model, train_iterator, optimizer, criterion)
            
            # Evaluate the model
            valid_loss, valid_acc, valid_f1 = evaluate(model, valid_iterator, criterion)
            
            # Save the best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), myargs['modelpath'])
            print(f'Epoch: {epoch}')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f} | Train F1: {train_f1:.2f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f} | Valid F1: {valid_f1:.2f}')
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/validation', valid_loss, epoch)
            writer.add_scalar('F1/train', train_f1, epoch)
            writer.add_scalar('F1/validation', valid_f1, epoch)
    
    else:
        pass

    writer.close()