import sys
import torch
from tqdm import tqdm
from loguru import logger
from typing import Union
from torch.optim import Adam
from src import EMBEDDING_PATH
from src.data.tokenizer import Tokenizer
from src.data.dataloader import ContinuousDataLoader, ShuffledDataLoader
from src.utils import readData, readEmbedding, getEmbeddingPath
from src.models.LSTM import LSTMModel

def train(model: torch.nn.Module, trainDataLoader: Union[ContinuousDataLoader, ShuffledDataLoader], optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, tokenizer: Tokenizer, batchCase: int, numEpochs: int):
    for epoch in range(numEpochs):
        predict(model, tokenizer)
        model.train()
        if batchCase == 3:
            h, c = model.initHidden(trainDataLoader.batchSize)
        with tqdm(total=len(trainDataLoader)) as pbar:
            pbar.set_description(f"Epoch {epoch + 1}")
            for batch in trainDataLoader:
                optimizer.zero_grad()
                x, y = batch
                if batchCase != 3:
                    h, c = model.initHidden(x.shape[0])
                output, h, c = model(x, h, c)
                loss: torch.Tensor = criterion(output.view(-1, len(tokenizer)), y.view(-1))
                loss.backward()
                optimizer.step()
                # logger.info(f"Loss: {loss.item()}")
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
                
    predict(model, tokenizer)

def predict(model: torch.nn.Module, tokenizer: Tokenizer):
    model.eval()
    text = "the magazine will reward with"
    while True:
        if text.split()[-1] == "<eos>" or len(text.split()) > 15:
            break
        input = torch.tensor(tokenizer(text.lower()), dtype=torch.long)[None, :]
        h, c = model.initHidden(input.shape[0])
        output, h, c = model(input, h, c)
        logit = output[0, -1, :]
        prediction = torch.argmax(logit).item()
        text += ' ' + tokenizer.detokenize(prediction)
    logger.info(f"Prediction: {text}")

def main():
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(f"log/lstmTest" + "_{time:YYYY-MM-DD:HH:mm:ss}.log", rotation="500 MB", level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    
    batchCase = 1 # Use shuffled batching, and set the first hidden state as a zero vector.
    # batchCase = 2 # Use continuous batching, and set the first hidden state as a zero vector.
    # batchCase = 3 # Use continuous batching, and use the last hidden state of the previous batch as the first hidden state of the current batch.
    
    batchSize = 4
    sequenceLength = 128
    embeddingDim = 50
    hiddenDim = 256
    numLayers = 4
    learningRate = 0.0005
    numEpochs = 10
    embeddingScale = 6
    
    trainPath = "data/penn-treebank/ptb.train.txt"
    validPath = "data/penn-treebank/ptb.valid.txt"
    testPath = "data/penn-treebank/ptb.test.txt"
    trainData = readData(trainPath)
    validData = readData(validPath)
    testData = readData(testPath)
    logger.info(f"Number of training tokens: {len(trainData)}")
    logger.info(f"Number of validation tokens: {len(validData)}")
    logger.info(f"Number of testing tokens: {len(testData)}")
    tokenizer = Tokenizer([trainData, validData, testData])
    logger.info(f"Vocabulary size: {len(tokenizer)}")

    tokenizedTrainData = tokenizer(trainData)
    tokenizedValidData = tokenizer(validData)
    tokenizedTestData = tokenizer(testData)
    logger.success("Tokenization finished")

    if batchCase == 1:
        trainDataLoader = ContinuousDataLoader(tokenizedTrainData, batchSize, sequenceLength, tokenizer.eosToken)
        validDataLoader = ContinuousDataLoader(tokenizedValidData, batchSize, sequenceLength, tokenizer.eosToken)
        testDataLoader = ContinuousDataLoader(tokenizedTestData, batchSize, sequenceLength, tokenizer.eosToken)
    else:
        trainDataLoader = ShuffledDataLoader(tokenizedTrainData, batchSize, sequenceLength)
        validDataLoader = ShuffledDataLoader(tokenizedValidData, batchSize, sequenceLength)
        testDataLoader = ShuffledDataLoader(tokenizedTestData, batchSize, sequenceLength)
    
    model = LSTMModel(len(tokenizer), embeddingDim, hiddenDim, numLayers)
    logger.info(model)
    embeddings = readEmbedding(getEmbeddingPath(embeddingScale, embeddingDim), tokenizer)
    logger.success("Embedding read")
    model.loadEmbedding(embeddings)
    logger.success("Embedding loaded")
    
    optimizer = Adam(model.parameters(), lr=learningRate)
    criterion = torch.nn.CrossEntropyLoss()
    
    train(model, trainDataLoader, optimizer, criterion, tokenizer, batchCase, numEpochs)
    
if __name__ == "__main__":
    main()