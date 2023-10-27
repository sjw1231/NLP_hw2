import torch
import torch.nn as nn
from typing import Dict

class RNNModel(nn.Module):
    def __init__(self, vocabSize, embeddingDim, hiddenDim, numLayers):
        super().__init__()
        self.vocabSize = vocabSize
        self.embeddingDim = embeddingDim
        self.hiddenDim = hiddenDim
        self.numLayers = numLayers
        self.device: torch.device
        
        self.embedding = nn.Embedding(vocabSize, embeddingDim)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.Tanh()
        self.rnn = nn.RNN(embeddingDim, hiddenDim, numLayers, batch_first=True)
        self.linear = nn.Linear(hiddenDim, vocabSize)
        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.zeros_(self.linear.weight.data)
        nn.init.zeros_(self.linear.bias.data)
        
    def loadEmbedding(self, embeddings: Dict[int, list]):
        for id, embedding in embeddings.items():
            self.embedding.weight.data[id] = torch.tensor(embedding)
            
    def forward(self, x, h):
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.activation(x)
        x, h = self.rnn(x, h)
        x = self.linear(x)
        return x, h
    
    def initHidden(self, batchSize):
        return torch.zeros(self.numLayers, batchSize, self.hiddenDim, device=self.device)