import torch
import torch.nn as nn
import torch.nn.functional as F

from torchcrf import CRF
from math import sqrt

SEED = 2020
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SentenceTagger(nn.Module):
    
    def __init__(self, params, glove_vectors):
        
        super(SentenceTagger, self).__init__()

        self.params = params
        self.word_embedding = nn.Embedding.from_pretrained(glove_vectors, freeze=True)
        self.char_embedding = nn.Embedding(params['num_char_embeddings'], params['char_embedding_dim'])
        num = sqrt(3/params['char_embedding_dim'])
        self.char_embedding.weight.data.uniform_(-num, num)
        

        self.conv = nn.Conv1d(params['char_embedding_dim'], params['cnn_num_filters'], 
                                params['cnn_window_size'], bias=True)

        self.dropout = nn.Dropout(params['dropout_rate'])

        self.lstm = nn.LSTM(params['lstm_embedding_size'], params['lstm_hidden_size'],
                            batch_first=True, bidirectional=True)

        self.fc = nn.Linear(2 * params['lstm_hidden_size'], params['number_of_tags'])

        self.crf = CRF(params['number_of_tags'],batch_first=True)

    def forward(self, input):

        # input.transpose_(0,1)
        word_embed = self.word_embedding(input[0]) # dim: batch_size x seq_len x embedding_dim

        char_embed = self.char_embedding(input[1])
        
        char_embed = self.dropout(char_embed)
    
        char_embed = char_embed.permute(0, 1, 3, 2)
        conv_shape = (char_embed.shape[0] * char_embed.shape[1],
                        char_embed.shape[2],
                        char_embed.shape[3])
        conv_input = char_embed.reshape(conv_shape)
        conv_output = self.conv(conv_input)
        conv_output = conv_output.max(dim=2)[0]
        conv_output = conv_output.reshape((char_embed.shape[0],
                                            char_embed.shape[1],
                                            conv_output.shape[1]))
        conv_output = conv_output.permute(1, 0, 2)
        word_embed = word_embed.float()
        conv_output = conv_output.float()
        conv_output = conv_output.permute(1, 0, 2)
        
        embedding = torch.cat((word_embed, conv_output), -1)

        embedding = self.dropout(embedding)
        # run the LSTM along the sentences of length seq_len
        s, _ = self.lstm(embedding)              # dim: batch_size x seq_len x lstm_hidden_dim

        s = self.dropout(s)

        # make the Variable contiguous in memory (a PyTorch artefact)
        s = s.contiguous()

        # reshape the Variable so that each row contains one token
        # s = s.view(-1, s.shape[2])       # dim: batch_size*seq_len x lstm_hidden_dim
        
        # apply the fully connected layer and obtain the output (before softmax) for each token
        s = self.fc(s)                   # dim: batch_size*seq_len x num_tags

        loss = -self.crf(s, input[2])
        out = torch.FloatTensor(self.crf.decode(s))
        return loss, out
        # apply log softmax on each token's output (this is recommended over applying softmax
        # since it is numerically more stable)
        # return F.log_softmax(s, dim=1)   # dim: batch_size*seq_len x num_tags
        return s