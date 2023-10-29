import os
import sys
import torch
import random
import torch.nn as nn
from tqdm import tqdm
from loguru import logger
from typing import Union, List
from argparse import ArgumentParser
from torch.optim import Adam
import matplotlib.pyplot as plt
from src.data.tokenizer import Tokenizer
from src.data.dataloader import ContinuousDataLoader, ShuffledDataLoader
from src.utils import readData, readEmbedding, getEmbeddingPath, calculatePPL
from src.models import LSTMModel, RNNModel, LockedDropoutLSTMModel

def train(model: LSTMModel, trainDataLoader: Union[ContinuousDataLoader, ShuffledDataLoader], validDataLoader: Union[ContinuousDataLoader, ShuffledDataLoader], optimizer: torch.optim.Optimizer, criterion: torch.nn.Module, tokenizer: Tokenizer, batchCase: int, numEpochs: int, modelName: str):
    # modelName = 'rnn' if useRNN else 'lstm'
    bestValidPPL = float("inf")
    
    trainLossList = []
    trainPPLList = []
    validLossList = []
    validPPLList = []
    
    for epoch in range(numEpochs):
        predict(model, tokenizer, "you want to eat a")
        model.train()
        lossList = []
        pplList = []
        if batchCase == 3:
            h = model.initHidden(trainDataLoader.batchSize)
        with tqdm(total=len(trainDataLoader)) as pbar:
            pbar.set_description(f"Epoch {epoch + 1}")
            for batch in trainDataLoader:
                optimizer.zero_grad()
                x, y = batch
                x = x.to(model.device)
                y = y.to(model.device)
                if batchCase != 3:
                    h = model.initHidden(trainDataLoader.batchSize)
                if modelName == 'rnn':
                    h = h.detach()
                elif modelName == 'lstm':
                    h = (h[0].detach(), h[1].detach())
                else:
                    h = [(hlayer[0].detach(), hlayer[1].detach()) for hlayer in h]
                output, h = model(x, h)
                loss: torch.Tensor = criterion(output.view(-1, len(tokenizer)), y.view(-1))
                loss.backward()
                optimizer.step()
                lossList.append(loss.item())
                
                ppl = calculatePPL(output, y)
                pplList.append(ppl.detach().cpu())
                
                pbar.update(1)
                pbar.set_postfix(loss=loss.item(), ppl=ppl.mean().item())
                
        logger.info(f"Average training loss: {sum(lossList) / len(lossList)}")
        trainLossList.append(sum(lossList) / len(lossList))
        pplList = torch.cat(pplList)
        logger.info(f"Average training ppl: {torch.mean(pplList).item()}")
        trainPPLList.append(torch.mean(pplList).item())
        
        validLoss, validPPL = evaluate(model, validDataLoader, criterion, tokenizer, batchCase)
        validLossList.append(validLoss)
        validPPLList.append(validPPL)
        if validPPL < bestValidPPL:
            bestValidPPL = validPPL
            torch.save(model.state_dict(), f"checkpoint/{modelName}_case{batchCase}.best.pt")
            logger.success("Model saved")
        
    # draw loss and ppl curve
    plt.title("Loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(range(1, numEpochs + 1), trainLossList, label="train", color='blue')
    plt.plot(range(1, numEpochs + 1), validLossList, label="valid", color='red')
    plt.legend()
    plt.savefig(f"img/{modelName}_case{batchCase}loss.png")
    plt.show()
    
    plt.clf()
    plt.title("Perplexity")
    plt.xlabel("epoch")
    plt.ylabel("ppl")
    plt.plot(range(1, numEpochs + 1), trainPPLList, label="train", color='blue')
    plt.plot(range(1, numEpochs + 1), validPPLList, label="valid", color='red')
    plt.legend()
    plt.savefig(f"img/{modelName}_case{batchCase}ppl.png")
    plt.show()
    
    predict(model, tokenizer, "you want to eat a")

@torch.no_grad()
def evaluate(model: LSTMModel, dataLoader: Union[ContinuousDataLoader, ShuffledDataLoader], criterion: torch.nn.Module, tokenizer: Tokenizer, batchCase: int, test: bool = False):
    model.eval()
    
    lossList = []
    pplList = []
    if batchCase == 3:
        h = model.initHidden(dataLoader.batchSize)
    with tqdm(total=len(dataLoader)) as pbar:
        if test:
            pbar.set_description(f"Testing")
        else:
            pbar.set_description(f"Validation")
        for batch in dataLoader:
            x, y = batch
            x = x.to(model.device)
            y = y.to(model.device)
            if batchCase != 3:
                h = model.initHidden(x.shape[0])
            output, h = model(x, h)
            loss: torch.Tensor = criterion(output.view(-1, len(tokenizer)), y.view(-1))
            lossList.append(loss.item())
            
            ppl = calculatePPL(output, y)
            pplList.append(ppl.detach().cpu())
            
            pbar.update(1)
            pbar.set_postfix(loss=loss.item(), ppl=ppl.mean().item())
            
    meanLoss = sum(lossList) / len(lossList)
    pplList = torch.cat(pplList)
    meanPPL = torch.mean(pplList).item()
    
    if test:
        logger.info(f"Average testing loss: {meanLoss}")
        logger.info(f"Average testing ppl: {meanPPL}")
        return meanLoss, meanPPL
    
    logger.info(f"Average validation loss: {meanLoss}")
    logger.info(f"Average validation ppl: {meanPPL}")
    return meanLoss, meanPPL

@torch.no_grad()
def predict(model: Union[LSTMModel, RNNModel], tokenizer: Tokenizer, text: str = None):
    model.eval()
    
    while True:
        if text.split()[-1] == "<eos>" or len(text.split()) > 15:
            break
        input = torch.tensor(tokenizer(text.lower()), dtype=torch.long)[None, :]
        input = input.to(model.device)
        h = model.initHidden(input.shape[0])
        output, h = model(input, h)
        logit = output[0, -1, :]
        prediction = torch.argmax(logit).item()
        text += ' ' + tokenizer.detokenize(prediction)
    logger.info(f"Prediction: {text}")

@torch.no_grad()  
def generateOneSentence(model: Union[LSTMModel, RNNModel], tokenizer: Tokenizer, text: str, lenSen: int):
    model.eval()
    sentence = tokenizer(text)
    while True:
        if sentence[-1] == tokenizer.eosToken or len(sentence) > lenSen:
            break
        input = torch.tensor(sentence, dtype=torch.long)[None, :]
        input = input.to(model.device)
        h = model.initHidden(input.shape[0])
        output, h = model(input, h)
        logits = output[0, -1, :]
        value, indice = torch.topk(logits, 40)
        value = torch.softmax(value, dim=0)
        predictID = torch.multinomial(value, 1).item()
        sentence.append(indice[predictID].item())
        
    sentence = sentence[:-1] if sentence[-1] == tokenizer.eosToken else sentence
    text = tokenizer.detokenize(sentence)
    return text

@torch.no_grad()
def generate(model: Union[LSTMModel, RNNModel], criterion: torch.nn.Module, tokenizer: Tokenizer, testPath: str):
    model.eval()
    with open(testPath, 'r') as f:
        testData = f.read().split('\n')
    testData = [sentence.split() for sentence in testData]
    testData = [sentence[:5] for sentence in testData if len(sentence) >= 5]
    testData = [' '.join(sentence) for sentence in testData]
    testData = [tokenizer(sentence) for sentence in testData]
    
    genData = []
    lossList = []
    pplList = []
    for sentence in tqdm(testData):
        while True:
            if sentence[-1] == tokenizer.eosToken or len(sentence) >= 20:
                break
            input = torch.tensor(sentence, dtype=torch.long)[None, :]
            input = input.to(model.device)
            h = model.initHidden(input.shape[0])
            output, h = model(input, h)
            logits = output[0, -1, :]
            value, indice = torch.topk(logits, 40)
            value = torch.softmax(value, dim=0)
            # random sample from `indice` with probabilities `value`
            predictID = torch.multinomial(value, 1).item()
            sentence.append(indice[predictID].item())
        sentence = sentence[:-1] if sentence[-1] == tokenizer.eosToken else sentence
        
        h = model.initHidden(1)
        input = torch.tensor(sentence[:-1], dtype=torch.long)[None, :]
        input = input.to(model.device)
        output, h = model(input, h)
        target = torch.tensor(sentence[1:], dtype=torch.long)[None, :]
        target = target.to(model.device)
        loss: torch.Tensor = criterion(output.view(-1, len(tokenizer)), target.view(-1))
        lossList.append(loss.item())
        
        ppl = calculatePPL(output, target)
        pplList.append(ppl.detach().cpu())
        genData.append(tokenizer.detokenize(sentence))
    
    pplList = torch.cat(pplList)
    meanPPL = torch.mean(pplList).item()
    logger.info(f"Average generation loss: {sum(lossList) / len(lossList)}")
    logger.info(f"Average generation ppl: {meanPPL}")
    
    with open("data/penn-treebank/ptb.test.gen.txt", 'w') as f:
        f.write('\n'.join(genData))

def getParsedArgs():
    parser = ArgumentParser()
    parser.add_argument("--evalOnly", action="store_true", default=False)
    parser.add_argument("--modelName", type=str, choices=['rnn', 'lstm', 'locked'], default='lstm')
    parser.add_argument("--batchCase", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--batchSize", type=int, default=4)
    parser.add_argument("--sequenceLength", type=int, default=128)
    parser.add_argument("--embeddingDim", type=int, choices=[50, 100, 200, 300], default=300)
    parser.add_argument("--embeddingScale", type=int, choices=[6, 42], default=42)
    parser.add_argument("--hiddenDim", type=int, default=256)
    parser.add_argument("--numLayers", type=int, default=4)
    parser.add_argument("--learningRate", type=float, default=0.0005)
    parser.add_argument("--numEpochs", type=int, default=15)
    parser.add_argument("--dropoutw", type=float, default=0.1)
    parser.add_argument("--cudaID", type=int, choices=range(10), default=7)
    args = parser.parse_args()
    return args

def main():
    args = getParsedArgs()
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    logger.add(f"log/{args.modelName}Test" + "_{time:YYYY-MM-DD:HH:mm:ss}.log", rotation="500 MB", level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    
    logger.info(args)
    # # batchCase = 1 # Use shuffled batching, and set the first hidden state as a zero vector.
    # # batchCase = 2 # Use continuous batching, and set the first hidden state as a zero vector.
    # # batchCase = 3 # Use continuous batching, and use the last hidden state of the previous batch as the first hidden state of the current batch.
    
    evalOnly = args.evalOnly
    modelName = args.modelName
    
    batchCase = args.batchCase
    
    batchSize = args.batchSize
    sequenceLength = args.sequenceLength
    embeddingDim = args.embeddingDim
    embeddingScale = args.embeddingScale
    hiddenDim = args.hiddenDim
    numLayers = args.numLayers
    learningRate = args.learningRate
    numEpochs = args.numEpochs
    dropoutw = args.dropoutw
    cudaID = args.cudaID
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cudaID)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
        trainDataLoader = ContinuousDataLoader(tokenizedTrainData, batchSize, sequenceLength, tokenizer.eosToken, shuffle=True)
        validDataLoader = ContinuousDataLoader(tokenizedValidData, batchSize, sequenceLength, tokenizer.eosToken, shuffle=False)
        testDataLoader = ContinuousDataLoader(tokenizedTestData, batchSize, sequenceLength, tokenizer.eosToken, shuffle=False)
    else:
        trainDataLoader = ShuffledDataLoader(tokenizedTrainData, batchSize, sequenceLength, shuffle=True)
        validDataLoader = ShuffledDataLoader(tokenizedValidData, batchSize, sequenceLength, shuffle=False)
        testDataLoader = ShuffledDataLoader(tokenizedTestData, batchSize, sequenceLength, shuffle=False)
    if modelName == 'rnn':
        model = RNNModel(len(tokenizer), embeddingDim, hiddenDim, numLayers)
    elif modelName == 'lstm':
        model = LSTMModel(len(tokenizer), embeddingDim, hiddenDim, numLayers)
    else:
        model = LockedDropoutLSTMModel(len(tokenizer), embeddingDim, hiddenDim, numLayers, dropoutw)
    model.to(device)
    model.device = device
    logger.info(model)
    criterion = torch.nn.CrossEntropyLoss()
    
    if not evalOnly:
        embeddings = readEmbedding(getEmbeddingPath(embeddingScale, embeddingDim), tokenizer)
        logger.success("Embedding read")
        model.loadEmbedding(embeddings)
        logger.success("Embedding loaded")
        
        optimizer = Adam(model.parameters(), lr=learningRate)
        
        train(model, trainDataLoader, validDataLoader, optimizer, criterion, tokenizer, batchCase, numEpochs, modelName)
    
    model.load_state_dict(torch.load(f"checkpoint/{modelName}_case{batchCase}.best.pt"))
    # evaluate(model, testDataLoader, criterion, tokenizer, batchCase, test=True)
    # predict(model, tokenizer, "you want to eat a")
    # generate(model, criterion, tokenizer, testPath)
    while True:
        text = input("Input a sentence prefix: ")
        if text == 'exit':
            break
        print(generateOneSentence(model, tokenizer, text, 30))
    
if __name__ == "__main__":
    main()