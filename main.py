import os
import random
import json
import timeit
from argparse import ArgumentParser

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.anke.sentence_classifier import SentenceClassifier
from src.anke.data_loader import ClassificationDataset
from src.hovy.cnn_blstm_crf import SentenceTagger
from src.hovy.data_loader import TaggingDataset

from src.anke.train import train as train_classification
from src.anke.train import evaluate as evaluate_classification
from src.hovy.train import train as train_tagging
from src.hovy.train import evaluate as evaluate_tagging

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('-a', '--architecture', help='select model architecture to use', required=True, choices=['anke', 'mahovy', 'bert'], default='mahovy')
    parser.add_argument('-M', '--mode', help='select mode to run', required=True, choices=['train', 'test'])
    parser.add_argument('-m', '--model_path', help='provide save/load path for the model', required=True)
    parser.add_argument('-tr', '--train_data_path', help='provide path trainig data', required=False, default='data/tagging/openstax/train')
    parser.add_argument('-te', '--test_data_path', help='provide path to test data', required=False, default='data/tagging/openstax/val')
    parser.add_argument('-p', '--param_file_path', help='provide path to hyperparameter json file', required=False, default='utils/params_hovy_openstax.json')
    parser.add_argument('-s', '--summary', help='provide tensorboard summary file suffix', required=False, default='hovy_openstax')
    parser.add_argument('-e', '--embedding', help='Embeddings type (for anke)', required=False, choices=['glove', 'w2v'])

    myargs = vars(parser.parse_args())
    print(myargs)
    with open(myargs['param_file_path'], 'r') as f:
        params = json.load(f)

    # tensorboard summary writer
    writer = SummaryWriter(filename_suffix=myargs['summary'])

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    print("Loading data")
    train_data_path = myargs['train_data_path']
    valid_data_path = myargs['test_data_path']
    
    # Check for task
    if myargs['architecture'] == 'anke':
        if myargs['embedding'] == 'glove':
            embedding_type = 'glove'
            embeddings_path = 'utils/glove.6B.100d.txt'
        else:
            embedding_type = 'w2v'
            embeddings_path = 'utils/GoogleNews-vectors-negative300.bin'

        train_dataset = ClassificationDataset(train_data_path, embedding_type, embeddings_path)
        valid_dataset = ClassificationDataset(valid_data_path, embedding_type, embeddings_path)

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], num_workers=2)
        valid_loader = DataLoader(valid_dataset, batch_size=params['batch_size'], num_workers=2)

        # Initialize the pretrained embedding
        if embedding_type == 'w2v':
            word_vectors = train_dataset.w2v.vectors

            # Add vectors for <pad> and OOV

            # Randomly initialized vector for OOV
            word_vectors = np.vstack((word_vectors, np.random.rand(300)))
            # 0 vector for padding
            word_vectors = np.vstack((word_vectors, np.zeros(300)))
            
            # Convert to tensor
            word_vectors = torch.from_numpy(word_vectors)

        else:
            word_vectors = train_dataset.glove_matrix

        model = SentenceClassifier(params, word_vectors)
        
        # Define optimizer and loss
        if params['optimizer'] == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        if params['loss_fn'] == 'bce':
            criterion = nn.BCELoss()
        elif params['loss_fn'] == 'ce':
            criterion = loss_fn

        # Push to cuda if available
        print(device)
        model = model.to(device)

        if myargs['mode'] == 'train':

            # Model Architecture
            print(model)

            # No. of trianable parameters  
            print(f'The model has {count_parameters(model):,} trainable parameters')

            best_valid_loss = float('inf')

            print("Training Started")
            train_start_time = timeit.default_timer()
            for epoch in range(params['num_epochs']):

                epoch_start_time = timeit.default_timer()
                
                # Train the model
                model.train()
                train_iterator = iter(train_loader)
                train_loss, train_acc, train_f1 = train_classification(model, train_iterator, optimizer, criterion)
                
                model.eval()
                # Evaluate the model
                valid_iterator = iter(valid_loader)
                valid_loss, valid_acc, valid_f1 = evaluate_classification(model, valid_iterator, criterion)
                
                # Save the best model
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), myargs['model_path'])

                epoch_end_time = timeit.default_timer()

                print(f'Epoch: {epoch} | time taken: {epoch_end_time - epoch_start_time}')
                print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Train F1: {train_f1:.2f}')
                print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% | Valid F1: {valid_f1:.2f}')
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/validation', valid_loss, epoch)
                writer.add_scalar('F1/train', train_f1, epoch)
                writer.add_scalar('F1/validation', valid_f1, epoch)

            train_end_time = timeit.default_timer()
            print(f'Total training time: {train_end_time - train_start_time}')

        else:

            train_iterator = iter(train_loader)
            valid_iterator = iter(valid_loader)

            model.load_state_dict(torch.load(myargs['model_path']))
            model.eval()

            valid_loss, valid_acc, valid_f1 = evaluate_classification(model, valid_iterator, criterion)
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% | Valid F1: {valid_f1:.2f}')

    if myargs['architecture'] == 'mahovy':

        embeddings_path = 'utils/glove.6B.100d.txt'

        train_dataset = TaggingDataset(train_data_path, embeddings_path)
        valid_dataset = TaggingDataset(valid_data_path, embeddings_path)

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], num_workers=2)
        valid_loader = DataLoader(valid_dataset, batch_size=params['batch_size'], num_workers=2)

        # Initialize the pretrained embedding
        glove_vectors = train_dataset.glove_matrix

        # Instantiate the model
        model = SentenceTagger(params, glove_vectors)

        # Define optimizer and loss
        if params['optimizer'] == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        elif params['optimizer'] == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=params['momentum'], weight_decay=0.05)
        
        if params['loss_fn'] == 'bce':
            criterion = nn.BCELoss()
        elif params['loss_fn'] == 'ce':
            criterion = nn.CrossEntropyLoss()
        elif params['loss_fn'] == 'nll':
            criterion = nn.NLLLoss()

        # Push to cuda if available
        print(device)
        model = model.to(device)

        if myargs['mode'] == 'train':

            best_valid_loss = float('inf')

            # Model Architecture
            print(model)
                
            print(f'The model has {count_parameters(model):,} trainable parameters')

            print("Training Started")
            train_start_time = timeit.default_timer()
            for epoch in range(params['num_epochs']):
                
                epoch_start_time = timeit.default_timer()
                
                # Train the model
                model.train()
                train_iterator = iter(train_loader)
                train_loss, train_acc, train_f1 = train_tagging(model, train_iterator, optimizer, criterion)
                
                model.eval()
                # Evaluate the model
                valid_iterator = iter(valid_loader)
                valid_loss, valid_acc, valid_f1 = evaluate_tagging(model, valid_iterator, criterion)
                
                # Save the best model
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), myargs['model_path'])

                epoch_end_time = timeit.default_timer()

                print(f'Epoch: {epoch} | time taken: {epoch_end_time - epoch_start_time}')
                print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}% | Train F1: {train_f1:.2f}')
                print(f'\t Val. Loss: {valid_loss:.3f} | Val. Acc: {valid_acc*100:.2f}% | Val. F1: {valid_f1:.2f}')
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/validation', valid_loss, epoch)
                writer.add_scalar('F1/train', train_f1, epoch)
                writer.add_scalar('F1/validation', valid_f1, epoch)

            train_end_time = timeit.default_timer()
            print(f'Total training time: {train_end_time - train_start_time}')

        else:

            train_iterator = iter(train_loader)
            valid_iterator = iter(valid_loader)

            model.load_state_dict(torch.load(myargs['model_path']))
            model.eval()

            valid_loss, valid_acc, valid_f1 = evaluate_tagging(model, valid_iterator, criterion)
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}% | Valid F1: {valid_f1:.2f}')

    writer.close()