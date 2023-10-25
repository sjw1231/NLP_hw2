import torch
import random
import copy
# from torch.utils.data.dataloader import DataLoader
from typing import List

class ContinuousDataLoader():
    def __init__(self, data: List[int], batchSize: int, seqLen: int, eosToken: int = 0):
        # super().__init__(dataset=data, batch_size=batchSize, shuffle=False, drop_last=True)
        self.data = data
        self.batchSize = batchSize
        self.seqLen = seqLen
        self.eosToken = eosToken
        self.numBatches = (len(data) - 1) // (batchSize * seqLen)
        self.dataSentence = []
        sentence = []
        for token in data:
            sentence.append(token)
            if token == eosToken:
                self.dataSentence.append(sentence)
                sentence = []
        
    def __iter__(self):
        dataSentence = copy.deepcopy(self.dataSentence)
        random.shuffle(dataSentence)
        data = [token for sentence in dataSentence for token in sentence]
        data = data[: self.numBatches * self.batchSize * self.seqLen + 1]
        lenSen = self.seqLen * self.numBatches
        
        for i in range(self.numBatches):
            # Batch shape [batchSize, seqLen + 1]. The last token is the first token of the next batch. Split data into batchSize sentences. Split each sentence into numBatches samples. Batch k consists of the k-th sample of each sentence.
            input = []
            output = []
            for j in range(self.batchSize):
                input.append(data[i * self.seqLen + j * lenSen : (i + 1) * self.seqLen + j * lenSen])
                output.append(data[i * self.seqLen + j * lenSen + 1 : (i + 1) * self.seqLen + j * lenSen + 1])
            input = torch.tensor(input, dtype=torch.long)
            output = torch.tensor(output, dtype=torch.long)
            yield input, output
            
    def __len__(self):
        return self.numBatches