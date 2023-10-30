import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser
from loguru import logger
from typing import List, Union, Tuple
from ..utils import readData, readEmbedding, getEmbeddingPath, calculatePPL
from ..data.tokenizer import Tokenizer
from ..data.dataloader import *
from ..models import *

class RNNTrainer():
    @staticmethod
    def get_parser(parser: ArgumentParser = None):
        if parser is None: parser = ArgumentParser()
        parser.add_argument("--verbose", type=str, choices=['DEBUG', 'INFO', 'SUCCESS', 'WARNING', 'ERROR', 'CRITICAL'], default='INFO')
        parser.add_argument("--mode", type=str, choices=['train', 'eval', 'generate', 'test'], default='train')
        parser.add_argument("--modelName", type=str, choices=['rnn', 'lstm', 'locked'], default='lstm')
        parser.add_argument("--activate", type=str, choices=['tanh', 'leakyrelu', 'gelu'],default='tanh')
        parser.add_argument("--batchCase", type=int, choices=[1, 2, 3], default=3)
        parser.add_argument("--batchSize", type=int, default=4)
        parser.add_argument("--sequenceLength", type=int, default=128)
        parser.add_argument("--embeddingDim", type=int, choices=[50, 100, 200, 300], default=300)
        parser.add_argument("--embeddingScale", type=int, choices=[6, 42], default=42)
        parser.add_argument("--hiddenDim", type=int, default=512)
        parser.add_argument("--numLayers", type=int, default=1)
        parser.add_argument("--learningRate", type=float, default=0.0005)
        parser.add_argument("--numEpochs", type=int, default=1000)
        parser.add_argument("--dropoutw", type=float, default=0.1)
        parser.add_argument("--useLrDecay", action='store_true', default=False)
        import py3nvml
        parser.add_argument("--cudaID", type=int, choices=range(len(py3nvml.get_num_procs())), default=7)
        parser.add_argument("--expName", type=str, default="0")
        return parser
    
    def initLogger(self, verbose: str):
        logger.remove()
        logger.add(sys.stderr, level=verbose)
        logger.add(f"log/{self.modelName}" + "_{time:YYYY-MM-DD:HH:mm:ss}.log", rotation="500 MB", level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
        
    def __init__(self, args):
        self.mode: str = args.mode
        self.modelName: str = args.modelName
        
        self.initLogger(args.verbose)
        logger.info(args)
        
        self.activate: str = args.activate
        self.batchCase: int = args.batchCase
        self.batchSize: int = args.batchSize
        self.sequenceLength: int = args.sequenceLength
        self.embeddingDim: int = args.embeddingDim
        self.embeddingScale: int = args.embeddingScale
        self.hiddenDim: int = args.hiddenDim
        self.numLayers: int = args.numLayers
        self.learningRate: float = args.learningRate
        self.numEpochs: int = args.numEpochs
        self.dropoutw: float = args.dropoutw
        self.cudaID: int = args.cudaID
        self.expName: str = args.expName
        self.useLrDecay: bool = args.useLrDecay
        self.device = torch.device('cuda:{}'.format(self.cudaID)) if torch.cuda.is_available() else torch.device('cpu')
        
        self.dataPath = "data/penn-treebank/ptb.{}.txt"
        self.trainData: List[str]
        self.validData: List[str]
        self.testData: List[str]
        self.trainDataLoader: Union[ContinuousDataLoader, ShuffledDataLoader]
        self.validDataLoader: Union[ContinuousDataLoader, ShuffledDataLoader]
        self.testDataLoader: Union[ContinuousDataLoader, ShuffledDataLoader]
        
    def readDataSplit(self, split: str):
        setattr(self, '{}Data'.format(split), readData(self.dataPath.format(split)))
        logger.info(f"Number of {split} tokens: {len(getattr(self, '{}Data'.format(split)))}")
        return getattr(self, '{}Data'.format(split))
        
    def getTokenizer(self):
        self.tokenizer = Tokenizer([self.trainData, self.validData, self.testData])
        return self.tokenizer
    
    def getDataLoader(self, split: str):
        tokenizedData = self.tokenizer(getattr(self, '{}Data'.format(split)))
        if self.batchCase == 1:
            dataLoader = ContinuousDataLoader(tokenizedData, self.batchSize, self.sequenceLength, self.tokenizer.eosToken, shuffle=True if split == 'train' else False)
        else:
            dataLoader = ShuffledDataLoader(tokenizedData, self.batchSize, self.sequenceLength, shuffle=True if split == 'train' else False)
        setattr(self, '{}DataLoader'.format(split), dataLoader)
        return dataLoader
    
    def getModel(self):
        if self.modelName == 'rnn':
            model = RNNModel(self.tokenizer.vocabSize, self.embeddingDim, self.hiddenDim, self.numLayers)
        elif self.modelName == 'lstm':
            model = LSTMModel(self.tokenizer.vocabSize, self.embeddingDim, self.hiddenDim, self.numLayers, self.activate)
        elif self.modelName == 'locked':
            model = LockedDropoutLSTMModel(self.tokenizer.vocabSize, self.embeddingDim, self.hiddenDim, self.numLayers, self.dropoutw)
        else:
            raise NotImplementedError
        
        model.to(self.device)
        model.device = self.device
        logger.info(model)
        
        if self.mode == 'train':
            embeddings = readEmbedding(getEmbeddingPath(self.embeddingScale, self.embeddingDim), self.tokenizer)
            logger.success(f"Embedding read from {getEmbeddingPath(self.embeddingScale, self.embeddingDim)}")
            model.loadEmbedding(embeddings)
            logger.success(f"Embedding loaded")
            
        self.model = model
        return model
    
    def getOptimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate)
        return self.optimizer
    
    def getCriterion(self):
        self.criterion = nn.CrossEntropyLoss()
        return self.criterion
    
    def getLrScheduler(self):
        # use lambda function to decay learning rate with warmup at the first 2 epochs
        # self.lrScheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 0.25 * 2 ** epoch if epoch < 2 else 0.95 ** (epoch - 2))
        self.lrScheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1 if epoch < 4 else 0.8 ** (epoch - 4))
        return self.lrScheduler
    
    def step(self, stage: str, batch: Tuple[torch.Tensor, torch.Tensor]):
        if stage == 'train':
            self.optimizer.zero_grad()
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        if self.batchCase != 3:
            h = self.model.initHidden(self.batchSize)
        else:
            h = self.lastHidden
        if self.modelName == 'rnn':
            h = h.detach()
        elif self.modelName == 'lstm':
            h = (h[0].detach(), h[1].detach())
        elif self.modelName == 'locked':
            h = [(hlayer[0].detach(), hlayer[1].detach()) for hlayer in h]
        else:
            raise NotImplementedError
        
        output, h = self.model(x, h)
        self.lastHidden = h
        loss: torch.Tensor = self.criterion(output.view(-1, self.tokenizer.vocabSize), y.view(-1))
        if stage == 'train':
            loss.backward()
            self.optimizer.step()
        
        ppl = calculatePPL(output, y)
        return loss.item(), ppl.detach().cpu()       
    
    def trainEpoch(self, epoch: int):
        self.model.train()
        lossList = []
        pplList = []
        if self.batchCase == 3:
            h = self.model.initHidden(self.batchSize)
        with tqdm(total=len(self.trainDataLoader)) as pbar:
            pbar.set_description(f"Epoch {epoch + 1}")
            self.lastHidden = self.model.initHidden(self.batchSize)
            for batch in self.trainDataLoader:
                loss, ppl = self.step('train', batch)
                lossList.append(loss)
                pplList.append(ppl)
                pbar.set_postfix(loss=loss, ppl=ppl.mean().item())
                pbar.update(1)
        
        loss = np.mean(lossList)
        ppl = torch.cat(pplList).mean().item()
        return loss, ppl
    
    def getSaveName(self):
        return f"{self.expName}/{self.modelName}_layers{self.numLayers}_act{self.activate}_case{self.batchCase}"
    
    def trainEpochEnd(self, epoch: int, loss: float, ppl: float):
        if self.mode == 'train' and self.useLrDecay:
            self.lrScheduler.step()
        logger.info(f"Epoch {epoch + 1} train loss: {loss}, train ppl: {ppl}")
        validLoss, validPPL = self.evaluate('valid')

        if validPPL < self.bestPPL:
            self.bestPPL = validPPL
            self.bestEpoch = epoch + 1
            torch.save(self.model.state_dict(), f"checkpoint/{self.getSaveName()}.best.pt")
            logger.success(f"Best model saved at epoch {self.bestEpoch}, with valid ppl {self.bestPPL}")
        
        return validLoss, validPPL
    
    @torch.no_grad()
    def evaluate(self, split: str):
        if split == 'valid':
            dataLoader = self.validDataLoader
        elif split == 'test':
            dataLoader = self.testDataLoader
        else:
            raise NotImplementedError
        
        self.model.eval()
        lossList = []
        pplList = []
        if self.batchCase == 3:
            h = self.model.initHidden(self.batchSize)
        with tqdm(total=len(dataLoader)) as pbar:
            if split == 'valid':
                pbar.set_description(f"Valid")
            elif split == 'test':
                pbar.set_description(f"Test")
            self.lastHidden = self.model.initHidden(self.batchSize)
            for batch in dataLoader:
                loss, ppl = self.step('eval', batch)
                lossList.append(loss)
                pplList.append(ppl)
                pbar.set_postfix(loss=loss, ppl=ppl.mean().item())
                pbar.update(1)
        
        loss = np.mean(lossList)
        ppl = torch.cat(pplList).mean().item()
        logger.info(f"{split} loss: {loss}, {split} ppl: {ppl}")
        return loss, ppl
        
    def drawCurve(self, trainMetricList: List[float], validMetricList: List[float], metricName: str):
        plt.figure()
        plt.title(f"{metricName} curve")
        plt.xlabel("Epoch")
        plt.ylabel(metricName)
        plt.plot(range(1, self.trueEpochs + 1), trainMetricList, label=f"train", color='blue')
        plt.plot(range(1, self.trueEpochs + 1), validMetricList, label=f"valid", color='red')
        plt.legend()
        plt.savefig(f"img/{self.getSaveName()}_{metricName}.png")
        plt.show()
    
    def fit(self):
        if not os.path.exists(f"output/{self.expName}"):
            os.mkdir(f"output/{self.expName}")
        if not os.path.exists(f"checkpoint/{self.expName}"):
            os.mkdir(f"checkpoint/{self.expName}")
        if not os.path.exists(f"img/{self.expName}"):
            os.mkdir(f"img/{self.expName}")
            
        self.model.train()
        self.bestPPL = float('inf')
        self.bestEpoch = 0
        
        trainLossList = []
        trainPPLList = []
        validLossList = []
        validPPLList = []
        
        for epoch in range(self.numEpochs):
            trainLoss, trainPPL = self.trainEpoch(epoch)
            trainLossList.append(trainLoss)
            trainPPLList.append(trainPPL)
            validLoss, validPPL = self.trainEpochEnd(epoch, trainLoss, trainPPL)
            validLossList.append(validLoss)
            validPPLList.append(validPPL)
            # early stopping
            if epoch - self.bestEpoch >= 10:
                self.trueEpochs = epoch + 1
                break
            
        with open(f"output/{self.getSaveName()}.json", 'w') as f:
            json.dump({
                'trainLoss': trainLossList,
                'trainPPL': trainPPLList,
                'validLoss': validLossList,
                'validPPL': validPPLList,
            }, fp=f, indent=4, ensure_ascii=False)
            
        self.drawCurve(trainLossList, validLossList, 'Loss')
        self.drawCurve(trainPPLList, validPPLList, 'Perplexity')
        
    def prepare(self):
        self.readDataSplit('train')
        self.readDataSplit('valid')
        self.readDataSplit('test')
        self.getTokenizer()
        self.getDataLoader('train')
        self.getDataLoader('valid')
        self.getDataLoader('test')
        self.getModel()
        self.getOptimizer()
        if self.useLrDecay:
            self.getLrScheduler()
        self.getCriterion()
    
    def run(self):
        self.prepare()
        if self.mode == 'train':
            self.fit()
            logger.success(f"Best model saved at epoch {self.bestEpoch}, with valid ppl {self.bestPPL}")
        self.model.load_state_dict(torch.load(f"checkpoint/{self.getSaveName()}.best.pt"))
        if self.mode == 'train' or self.mode == 'eval':
            loss, ppl = self.evaluate('test')
            logger.success(f"Test loss: {loss}, test ppl: {ppl}")
        elif self.mode == 'generate':
            length = int(input("Please input the length: "))
            while True:
                sentence = input("Please input a sentence: ")
                if sentence == 'exit':
                    break
                print(self.generate(sentence, length))
        elif self.mode == 'test':
            self.test()
        else:
            raise NotImplementedError
        logger.success(f"Experiment {self.expName} finished")
    
    @torch.no_grad()
    def generate(self, sentence: str, length: int, topk: int = 40):
        self.model.eval()
        sentence = self.tokenizer(sentence)
        sentence = torch.tensor(sentence, dtype=torch.long, device=self.device).unsqueeze(0)
        h = self.model.initHidden(1)
        output, h = self.model(sentence, h)
        for i in range(length):
            logits = output[0, -1, :]
            value, indice = torch.topk(logits, topk)
            value = torch.softmax(value, dim=0)
            predictID = torch.multinomial(value, 1).item()
            word = indice[predictID].item()
            
            if word == self.tokenizer.eosToken:
                break
            sentence = torch.cat([sentence, torch.tensor([[word]], dtype=torch.long, device=self.device)], dim=1)
            output, h = self.model(sentence[:, -1].unsqueeze(1), h)
        return self.tokenizer.detokenize(sentence[0].cpu().numpy().tolist())
    
    @torch.no_grad()
    def test(self):
        self.model.eval()
        with open(self.dataPath.format('test'), 'r') as f:
            testData = f.read().split('\n')
        testData = [sentence.split() for sentence in testData]
        testData = [sentence[:5] for sentence in testData if len(sentence) >= 5]
        testData = [' '.join(sentence) for sentence in testData]
        testData = [self.tokenizer(sentence) for sentence in testData]
        
        genData = []
        lossList = []
        pplList = []
        for sentence in tqdm(testData):
            while True:
                if sentence[-1] == self.tokenizer.eosToken or len(sentence) >= 20:
                    break
                input = torch.tensor(sentence, dtype=torch.long)[None, :]
                input = input.to(self.device)
                h = self.model.initHidden(input.shape[0])
                output, h = self.model(input, h)
                logits = output[0, -1, :]
                value, indice = torch.topk(logits, 40)
                value = torch.softmax(value, dim=0)
                # random sample from `indice` with probabilities `value`
                predictID = torch.multinomial(value, 1).item()
                sentence.append(indice[predictID].item())
            sentence = sentence[:-1] if sentence[-1] == self.tokenizer.eosToken else sentence
            
            h = self.model.initHidden(1)
            input = torch.tensor(sentence[:-1], dtype=torch.long)[None, :]
            input = input.to(self.model.device)
            output, h = self.model(input, h)
            target = torch.tensor(sentence[1:], dtype=torch.long)[None, :]
            target = target.to(self.model.device)
            loss: torch.Tensor = self.criterion(output.view(-1, len(self.tokenizer)), target.view(-1))
            lossList.append(loss.item())
            
            ppl = calculatePPL(output, target)
            pplList.append(ppl.detach().cpu())
            genData.append(self.tokenizer.detokenize(sentence))
        
        pplList = torch.cat(pplList)
        meanPPL = torch.mean(pplList).item()
        logger.info(f"Average generation loss: {sum(lossList) / len(lossList)}")
        logger.info(f"Average generation ppl: {meanPPL}")
        
        with open(self.dataPath.format(f'{self.expName}test.gen'), 'w') as f:
            f.write('\n'.join(genData))