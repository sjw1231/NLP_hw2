#! bin/bash

modelName=lstm
batchCase=3
numLayers=1
activate=tanh
dropoutw=0.1
useLrDecay=False

# check whether useLrDecay is true
if [ $useLrDecay = True ]
then
    python main.py --verbose=SUCCESS --cudaID=0 --mode=train --modelName=$modelName --activate=$activate --batchCase=$batchCase --numLayers=$numLayers --dropoutw=$dropoutw --useLrDecay --expName=train
else
    python main.py --verbose=SUCCESS --cudaID=0 --mode=train --modelName=$modelName --activate=$activate --batchCase=$batchCase --numLayers=$numLayers --dropoutw=$dropoutw --expName=train
fi