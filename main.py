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
    parser.add_argument('-tr', '--train_data_path', help='provide path trainig data', required=False, default='data/classification/wcl/train')
    parser.add_argument('-te', '--test_data_path', help='provide path to test data', required=False, default='data/classification/wcl/val')
    parser.add_argument('-p', '--param_file_path', help='provide path to hyperparameter json file', required=False, default='utils/params_anke_wcl.json')
    parser.add_argument('-c', '--is_csv', help='whether the data is in csv', required=False, choices=[True, False], default=False)


    # hyperparameters
    # parser.add_argument('-p', '--param_file_path', help='provide path to hyperparameter json file', required=False, default='utils/params_classification.json')

    myargs = vars(parser.parse_args())
    print(myargs)
    with open(myargs['param_file_path'], 'r') as f:
        params = json.load(f)

    # tensorboard summary writer
    writer = SummaryWriter()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')      
    # Load data
    print("Loading data")    
    train_data_path = myargs['train_data_path']
    valid_data_path = myargs['test_data_path']
    is_csv = myargs['is_csv']
    
    if myargs['mode'] == 'train':

        train_iterator, valid_iterator, TEXT, LABEL = load_data(train_data_path, valid_data_path, is_csv, params['batch_size'])

        # Define parameters
        params['num_embeddings'] = len(TEXT.vocab)
        with open(myargs['param_file_path'], 'w') as f:
            json.dump(params, f)

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
            model.train()
            train_loss, train_acc, train_f1 = train(model, train_iterator, optimizer, criterion)
            
            model.eval()
            # Evaluate the model
            valid_loss, valid_acc, valid_f1 = evaluate(model, valid_iterator, criterion)
            
            # Save the best model
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), myargs['modelpath'])
            print(f'Epoch: {epoch}')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Train F1: {train_f1:.2f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% | Valid F1: {valid_f1:.2f}')
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/validation', valid_loss, epoch)
            writer.add_scalar('F1/train', train_f1, epoch)
            writer.add_scalar('F1/validation', valid_f1, epoch)
    
    else:
        
        train_iterator, valid_iterator, TEXT, LABEL = load_data(train_data_path, valid_data_path, is_csv, params['batch_size'])

        # Initialize the pretrained embedding
        glove_vectors = TEXT.vocab.vectors

        if myargs['architecture'] == 'anke':
            model = SentenceClassifier(params, glove_vectors)
        model.load_state_dict(torch.load(myargs['modelpath']))
        model.eval()

        if params['loss_fn'] == 'bce':
            criterion = nn.BCELoss()

        # Push to cuda if available
        model = model.to(device)
        criterion = criterion.to(device)

        valid_loss, valid_acc, valid_f1 = evaluate(model, valid_iterator, criterion)
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% | Valid F1: {valid_f1:.2f}')

    writer.close()