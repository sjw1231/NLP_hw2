# Natural Language Processing, Assignment 2

## File structure
- `src/data/`
    - `tokenizer.py`: Class `Tokenizer` implements the tokenizer with functions about building vocabulary, tokenizing and detokenizing.
    - `dataloader/continuous_dataloader.py`: Class `ContinuousDataLoader` implements the dataloader for continuous batching method.
    - `dataloader/shuffled_dataloader.py`: Class `ShuffledDataLoader` implements the dataloader for shuffled batching method.
- `src/models/`
    - `LSTM.py`: Class `LSTMModel` implements the LSTM model with functions about forward propagation, backward propagation, initializing weight and hidden vectors and loading embeddings.
    - `RNN.py`: Class `RNNModel` implements the RNN model with functions about forward propagation, backward propagation, initializing weight and hidden vectors and loading embeddings.
    - `LockedDropoutLSTM.py`: Class `LockedDropoutLSTM` implements the LSTM model with locked dropout for Problem 3.
- `src/utils/utils.py`: Implements the functions about reading data, reading embeddings, getting embedding paths and calculating perplexity.
- `src/trainer/rnn_trainer.py`: Class `RNNTrainer` implements the trainer for RNN and LSTM model with functions about training, testing and generation.
- `scripts/`
    - `train.sh`: Implements the bash script for training.
    - `valid.sh`: Implements the bash script for generation.
- `draw.py`: Implements the function about drawing learning curves with comparison.
- `main.py`: Implements the main function.

## Script usage
Run `train.sh` for model training. Options include `modelName`(lstm, rnn or locked), `batchCase`(1, 2 or 3), `numLayers`, `activate`(tanh, leakyrelu or gelu), `dropoutw` and `useLrDecay`.
```
./scripts/train.sh
```
Run `valid.sh` for text generation. For option `mode`, 'test' is to generate 15 tokens on the test set, and 'generate' is to generate vaious length of sentences as required.
```
./scripts/valid.sh
```
