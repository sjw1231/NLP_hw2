import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loguru import logger
from typing import List, Dict

class WeightDrop(nn.Module):
    def __init__(self, module: nn.Module, weights: List[str], dropout: float):
        super().__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self._setup()
        
    def _setup(self):
        for name_w in self.weights:
            w = getattr(self.module, name_w)
            del self.module._parameters[name_w]
            self.module.register_parameter(name_w + '_raw', nn.Parameter(w.data))
            
    def _setweights(self):
        for name_w in self.weights:
            raw_w = getattr(self.module, name_w + '_raw')
            w = None
            mask = raw_w.new_ones((raw_w.size(0), 1))
            mask = F.dropout(mask, p=self.dropout, training=True)
            w = mask.expand_as(raw_w) * raw_w
            setattr(self.module, name_w, w)
            # logger.debug(f"After weightdrop: w is of shape {w.shape}")
            
    def forward(self, *args):
        self._setweights()
        return self.module.forward(*args)

class LockedDropoutLSTMModel(nn.Module):
    def __init__(self, vocabSize, embeddingDim, hiddenDim, numLayers, dropoutw):
        super().__init__()
        self.vocabSize = vocabSize
        self.embeddingDim = embeddingDim
        self.hiddenDim = hiddenDim
        self.numLayers = numLayers
        self.device: torch.device
    
        self.embedding = nn.Embedding(vocabSize, embeddingDim)
        self.embeddingDropout = nn.Dropout(0.1)
        self.activation = nn.Tanh()
        
        self.lstms = [nn.LSTM(input_size=embeddingDim if l == 0 else hiddenDim, hidden_size=hiddenDim, num_layers=1, batch_first=True) for l in range(numLayers)]
        if self.training:
            self.lstms = [WeightDrop(lstm, ['weight_hh_l0', 'weight_ih_l0'], dropoutw) for lstm in self.lstms]
        self.lstm = nn.ModuleList(self.lstms)
        
        self.linear = nn.Linear(hiddenDim, vocabSize)
        self.init_weights()
        
    def init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.zeros_(self.linear.weight.data)
        nn.init.zeros_(self.linear.bias.data)
    
    def loadEmbedding(self, embeddings: Dict[int, list]):
        for id, embedding in embeddings.items():
            self.embedding.weight.data[id] = torch.tensor(embedding)
    
    def forward(self, x, h: List[tuple]):
        x = self.embedding(x)
        x = self.embeddingDropout(x)
        x = self.activation(x)
        
        for l, layer in enumerate(self.lstm):
            x, h[l] = layer(x, h[l])
        x = self.linear(x)
        return x, h
    
    def initHidden(self, batchSize):
        return [(torch.zeros(1, batchSize, self.hiddenDim, device=self.device),
                torch.zeros(1, batchSize, self.hiddenDim, device=self.device)) for _ in range(self.numLayers)]