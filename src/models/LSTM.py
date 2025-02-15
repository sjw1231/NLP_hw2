import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict

class LSTMModel(nn.Module):
    # @staticmethod
    # def get_parser(parser=None):
    #     parser.add_argument("--embeddingDim", type=int, default=300)
    #     parser.add_argument("--hiddenDim", type=int, default=64)
    #     parser.add_argument("--numLayers", type=int, default=2)
    
    def __init__(self, vocabSize, embeddingDim, hiddenDim, numLayers, activate: str='tanh'):
        super().__init__()
        self.vocabSize = vocabSize
        self.embeddingDim = embeddingDim
        self.hiddenDim = hiddenDim
        self.numLayers = numLayers
        self.activate = activate
        self.device: torch.device
    
        self.embedding = nn.Embedding(vocabSize, embeddingDim)
        self.dropout = nn.Dropout(0.1)
        if activate == 'leakyrelu':
            self.activation = nn.LeakyReLU()
        elif activate == 'gelu':
            self.activation = nn.GELU()
        elif activate == 'tanh':
            self.activation = nn.Tanh()
        self.lstm = nn.LSTM(embeddingDim, hiddenDim, numLayers, batch_first=True)
        self.linear = nn.Linear(hiddenDim, vocabSize)
        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.zeros_(self.linear.weight.data)
        nn.init.zeros_(self.linear.bias.data)
    
    def loadEmbedding(self, embeddings: Dict[int, list]):
        for id, embedding in embeddings.items():
            self.embedding.weight.data[id] = torch.tensor(embedding)
    
    def forward(self, x, h: tuple):
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.activation(x)
        x, h = self.lstm(x, h)
        # h = (h_n, c_n). To deal with understanding tasks, we usually use h_n's last layer, which is h[0][:, -1, :]. For this generation task, since output is of same shape as input, we use x directly.
        x = self.linear(x)
        return x, h
    
    def initHidden(self, batchSize):
        return (torch.zeros(self.numLayers, batchSize, self.hiddenDim, device=self.device),
                torch.zeros(self.numLayers, batchSize, self.hiddenDim, device=self.device))