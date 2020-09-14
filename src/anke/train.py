from sklearn.metrics import f1_score

import torch

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
        
        batch[0] = batch[0].to(device)
        batch[1] = batch[1].to(device)
        # print(batch[0], batch[1])
        # print(type(batch[0]), type(batch[1]))
        #retrieve text and no. of words
        text = batch[0]
        
        #convert to 1D tensor
        predictions = model(text).squeeze()
        
        #compute the loss
        loss = criterion(predictions, batch[1].float())
        
        #compute the binary accuracy, f1
        acc = binary_accuracy(predictions, batch[1])
        f1 = f1_score(batch[1].data.to('cpu'), predictions.to('cpu') > 0.5, average="binary", zero_division=0)

        #backpropage the loss and compute the gradients
        loss.backward()

        #update the weights
        optimizer.step()
        
        #loss and accuracy
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        epoch_f1 += f1.item()
        
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
        
            batch[0] = batch[0].to('cuda')
            batch[1] = batch[1].to('cuda')

            #retrieve text and no. of words
            text = batch[0]
            
            #convert to 1d tensor
            predictions = model(text).squeeze()
            
            #compute loss, accuracy and f1
            loss = criterion(predictions, batch[1].float())
            acc = binary_accuracy(predictions, batch[1])
            f1 = f1_score(batch[1].data.to('cpu'), predictions.to('cpu') > 0.5, average="binary", zero_division=0)
            
            #keep track of loss and accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_f1 += f1.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator), epoch_f1 / len(iterator)

def predict(model, iterator):
    with torch.no_grad():
        for batch in iterator:
            batch[0] = batch[0].to('cuda')
            batch[1] = batch[1].to('cuda')

            text = batch[0]
            
            #convert to 1d tensor
            predictions = model(text).squeeze()
            
    return predictions

def binary_accuracy(preds, y):
    
    # Round predictions to the closest integer
    rounded_preds = torch.round(preds)
    
    correct = (rounded_preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc