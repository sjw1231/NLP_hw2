import json
import matplotlib.pyplot as plt
import numpy as np

trainLossDict = {}
validLossDict = {}
trainPPLDict = {}
validPPLDict = {}

filePath = 'output/2-1/lstm_layers{}_act{}_case3.json'

# minLen = int(1000)
# for layer in [1, 2, 4]:
#     with open(filePath.format(layer, 'tanh'), 'r') as f:
#         data = json.load(f)
#         minLen = min(minLen, len(data['trainLoss']))
#         trainLossDict[(layer, 'tanh')] = data['trainLoss']
#         validLossDict[(layer, 'tanh')] = data['validLoss']
#         trainPPLDict[(layer, 'tanh')] = data['trainPPL']
#         validPPLDict[(layer, 'tanh')] = data['validPPL']

# def draw_layers(dataDict, title, ylabel, filename):
#     plt.figure()
#     plt.title(title)
#     plt.xlabel('Epoch')
#     plt.ylabel(ylabel)
#     for layer in [1, 2, 4]:
#         plt.plot(range(1, minLen + 1), dataDict[(layer, 'tanh')][:minLen], label='{} layer'.format(layer))
#     plt.legend()
#     plt.savefig(filename)
#     plt.show()
    
# draw_layers(trainLossDict, 'Training loss', 'Loss', 'img/2-1/train_loss_layer.png')
# draw_layers(validLossDict, 'Validation loss', 'Loss', 'img/2-1/valid_loss_layer.png')
# draw_layers(trainPPLDict, 'Training perplexity', 'Perplexity', 'img/2-1/train_ppl_layer.png')
# draw_layers(validPPLDict, 'Validation perplexity', 'Perplexity', 'img/2-1/valid_ppl_layer.png')

minLen = int(1000)
for act in ['tanh', 'leakyrelu', 'gelu']:
    with open(filePath.format(1, act), 'r') as f:
        data = json.load(f)
        minLen = min(minLen, len(data['trainLoss']))
        trainLossDict[(1, act)] = data['trainLoss']
        validLossDict[(1, act)] = data['validLoss']
        trainPPLDict[(1, act)] = data['trainPPL']
        validPPLDict[(1, act)] = data['validPPL']

def draw_acts(dataDict, title, ylabel, filename):
    plt.figure()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    for act in ['tanh', 'leakyrelu', 'gelu']:
        plt.plot(range(1, minLen + 1), dataDict[(1, act)][:minLen], label='{} activation'.format(act))
    plt.legend()
    plt.savefig(filename)
    plt.show()

draw_acts(trainLossDict, 'Training loss', 'Loss', 'img/2-1/train_loss_act.png')
draw_acts(validLossDict, 'Validation loss', 'Loss', 'img/2-1/valid_loss_act.png')
draw_acts(trainPPLDict, 'Training perplexity', 'Perplexity', 'img/2-1/train_ppl_act.png')
draw_acts(validPPLDict, 'Validation perplexity', 'Perplexity', 'img/2-1/valid_ppl_act.png')