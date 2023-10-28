from loguru import logger
import torch
import random
import copy
# from torch.utils.data.dataloader import DataLoader
from typing import List

class ShuffledDataLoader():
    def __init__(self, data: List[int], batchSize: int, seqLen: int, shuffle: bool = True):
        self.data = data
        self.batchSize = batchSize
        self.seqLen = seqLen
        self.shuffle = shuffle
        self.numBatches = (len(data) - 1) // (batchSize * seqLen)
        
    def __iter__(self):
        dataSeq = []
        for i in range(self.numBatches * self.batchSize):
            dataSeq.append(self.data[i * self.seqLen : (i + 1) * self.seqLen + 1])
        # logger.debug(f"dataSeq: {len(dataSeq)} * {len(dataSeq[0])}")
        if self.shuffle:
            random.shuffle(dataSeq)
        
        for i in range(self.numBatches):
            input = []
            output = []
            for j in range(self.batchSize):
                input.append(dataSeq[i * self.batchSize + j][: -1])
                output.append(dataSeq[i * self.batchSize + j][1 :])
            input = torch.tensor(input, dtype=torch.long)
            output = torch.tensor(output, dtype=torch.long)
            yield input, output
            
    def __len__(self):
        return self.numBatches