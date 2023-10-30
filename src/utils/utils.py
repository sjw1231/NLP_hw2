import torch
import random
import numpy as np
from tqdm import tqdm
from ..data.tokenizer import Tokenizer
from .. import EMBEDDING_6B_PATH, EMBEDDING_42B_PATH

def readData(path: str):
    """
    Read data from `path`. Replace newline characters with `<eos>`.
    The format of the data is one sentence per line.
    :param path: path to the data file
    :return: data as a list of words
    """
    with open(path, "r") as f:
        data = f.read().replace("\n", "<eos>").split()
    return data

def readEmbedding(path: str, tokenizer: Tokenizer):
    """
    Read embedding from `path`. The format of the data is one word per line.
    :param path: path to the embedding file
    :param tokenizer: tokenizer to convert word to id
    :return: embeddings as a dictionary from word id to embedding
    """
    with open(path, "r") as f:
        embeddings = {}
        for line in tqdm(f):
            line = line.split()
            word = line[0]
            embedding = line[1:]
            if word not in tokenizer.vocab:
                continue
                
            id = tokenizer.word2idx[word]
            embedding = [float(e) for e in embedding]
            embeddings[id] = embedding
    return embeddings

def getEmbeddingPath(scale: int, embeddingDim: int):
    assert scale in [6, 42]
    if scale == 6:
        return EMBEDDING_6B_PATH.format(embeddingDim)
    else:
        return EMBEDDING_42B_PATH.format(embeddingDim)
    
def calculatePPL(output: torch.Tensor, target: torch.Tensor):
    probs = torch.softmax(output, dim=-1)
    logProbs = torch.log(probs) # [batchSize, sequenceLength, vocabSize]
    entropy = torch.gather(logProbs, dim=-1, index=target.unsqueeze(-1)).squeeze(-1) # [batchSize, sequenceLength]
    ppl = torch.exp(-entropy.mean(dim=-1)) # [batchSize]
    return ppl

def setSeeds(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)