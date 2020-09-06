import torch
import torch.nn as nn

SEED = 2020
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class SentenceClassifier(nn.Module):
    
    def __init__(self, params, w2v_vectors):
        
        super(SentenceClassifier, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(w2v_vectors, True)
        # self.embedding = nn.Embedding(params['num_embeddings'], params['embedding_dim'])
        
        self.conv = nn.Conv1d(params['embedding_dim'],
                             params['cnn_num_filters'],
                             params['cnn_kernel'],
                             padding=int(params['cnn_kernel']/2),
                             bias=True)

        nn.init.xavier_uniform_(self.conv.weight, gain=1)

        self.maxpool = nn.MaxPool1d(params['pool_kernel'], params['pool_stride'])

        self.dropout = nn.Dropout(params['dropout_rate'], inplace=True)

        self.lstm = nn.LSTM(params['cnn_num_filters'],
                            params['lstm_hidden_dim'],
                            batch_first=True,
                            bidirectional=True)

        # Change this layer with CRF later
        self.fc1 = nn.Linear(2*params['lstm_hidden_dim'], 1, bias=True)

    def forward(self, input):

        # input.transpose_(0,1)

        x = self.embedding(input)
        
        x = x.permute(0,2,1)
        x = self.conv(x.float())
        
        x = self.maxpool(x)
        
        x = self.dropout(x)
        
        x = x.permute(0,2,1)
        _, (x,__) = self.lstm(x)
        
        x = x.contiguous()
        x = x.permute(1,0,2)
        x = x.contiguous()
        x = x.view(-1, x.shape[2]*2)       
        x = self.dropout(x)
        
        x = self.fc1(x)
        
        x = torch.sigmoid(x)

        return x