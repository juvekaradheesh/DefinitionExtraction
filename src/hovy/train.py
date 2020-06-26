import os
import random
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
# from seqeval.metrics import f1_score

import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data

SEED = 2020
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(model, iterator, optimizer, criterion):
    
    #initialize every epoch 
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0
    
    #set the model in training phase
    model.train()  
    
    for batch in iterator:
        #resets the gradients after every batch
        optimizer.zero_grad()
        
        batch[0] = batch[0].to(device)#
        batch[1] = batch[1].to(device)
        batch[2] = batch[2].to(device)
        one_hot = nn.functional.one_hot(batch[2], model.params['number_of_tags'])
        one_hot = one_hot.view(-1, one_hot.shape[2]) # each row contains one token
        
        #convert to 1D tensor
        predictions = model(batch)

        _, preds = predictions.max(dim=1)
        _, labels = one_hot.max(dim=1)

        #compute the loss
        loss = criterion(predictions, labels)
        
        #compute the binary accuracy, f1
        acc = multiclass_accuracy(preds, labels)
        f1 = f1_score(labels.to('cpu'), preds.to('cpu'), labels = [0, 1, 3, 4], average="macro", zero_division=1)
        # labels 0 : B, 1 : I, 3 : E, 4 : S

        #backpropage the loss and compute the gradients
        loss.backward()

        # Gradient Clipping
        nn.utils.clip_grad_value_(model.parameters(), model.params['grad_clip'])

        #update the weights
        optimizer.step()
        
        #loss and accuracy
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_f1 += f1
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_f1 / len(iterator)

def evaluate(model, iterator, criterion):
    
    #initialize every epoch
    epoch_loss = 0
    epoch_acc = 0
    epoch_f1 = 0

    #deactivating dropout layers
    model.eval()
    
    #deactivates autograd
    with torch.no_grad():
    
        for batch in iterator:
        
            batch[0] = batch[0].to(device)
            batch[1] = batch[1].to(device)
            batch[2] = batch[2].to(device)
            one_hot = nn.functional.one_hot(batch[2], model.params['number_of_tags'])
            one_hot = one_hot.view(-1, one_hot.shape[2]) # flatten
            
            #convert to 1d tensor
            predictions = model(batch)

            _, labels = one_hot.max(dim=1)
            _, preds = predictions.max(dim=1)

            #compute loss, accuracy and f1
            loss = criterion(predictions, labels)
            acc = multiclass_accuracy(preds, labels)
            f1 = f1_score(labels.to('cpu'), preds.to('cpu'), labels = [0, 1, 3, 4], average="macro", zero_division=1)
            # labels 0 : B, 1 : I, 3 : E, 4 : S
            
            #keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_f1 += f1.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_f1 / len(iterator)

def multiclass_accuracy(preds, y):
    correct = (preds == y).float()
    acc = correct.sum()/ len(correct)
    return acc