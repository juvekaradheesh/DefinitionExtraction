import os
import random
import json
from sys import argv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data

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
    myargs = getopts(argv)
    if '-h' in myargs:
        print("\nUsage: python main.py -s save_model [optional] -t -m -d -p \n")
        print("\t-s: file path to save the model")
        print("\t-t: tasks ('tagging', 'calssification')")
        print("\t-m: models ('anke'[with classification],  'mahovy' [with tagging], 'tagging_bert' [with tagging])")
        print("\t-d: datsets ('wcl', 'w00', 'openstax', 'all')")
        print("\t-p: hyperparameters file path")
        exit()
        
    if not '-t' in myargs:
        print("\n\tMissing argument '-s'.\n")
        print("\tType 'python main.py -h' for more info.")
        exit()

    # Set default
    if not '-t' in myargs:
        myargs['-t'] = 'classification'
    
    if not '-m' in myargs:
        myargs['-m'] = 'anke'
    
    if not '-d' in myargs:
        myargs['-d'] = 'wcl'
    
    if not '-p' in myargs:
        myargs['-p'] = 'utils/params_classification.json'

    with open(myargs['-p']) as f:
        params = json.load(f)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

    # Load data
    print("Loading data")    
    if myargs['-d'] == 'all':
        def_data_dir = os.path.join('data', myargs['-t'], 'all')
        train_data_path = os.path.join(def_data_dir, 'train')
        valid_data_path = os.path.join(def_data_dir, 'val')
        is_csv = False
    
    if myargs['-d'] == 'wcl':
        def_data_dir = os.path.join('data', myargs['-t'], 'wcl')
        train_data_path = os.path.join(def_data_dir, 'train')
        valid_data_path = os.path.join(def_data_dir, 'val')
        is_csv = False

    if myargs['-d'] == 'w00':
        def_data_dir = os.path.join('data', myargs['-t'], 'w00')
        train_data_path = os.path.join(def_data_dir, 'train')
        valid_data_path = os.path.join(def_data_dir, 'val')
        is_csv = False
    

    train_iterator, valid_iterator, TEXT, LABEL = load_data(train_data_path, valid_data_path, is_csv, params['batch_size'])

    # Define parameters
    params['num_embeddings'] = len(TEXT.vocab)

    #Initialize the pretrained embedding
    glove_vectors = TEXT.vocab.vectors

    # Instantiate the model
    if myargs['-m'] == 'anke':
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
            torch.save(model.state_dict(), myargs['-s'])
        print(f'Epoch: {epoch}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f} | Train F1: {train_f1:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f} | Valid F1: {valid_f1:.2f}%')