from typing import List, Union

class Tokenizer:
    def __init__(self, dataList: List[List[str]]) -> None:
        self.dataList = dataList
        self.vocab = set()
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        self.build_vocab()

    def build_vocab(self) -> None:
        for sentence in self.dataList:
            for word in sentence:
                if word == 'N':
                    word = '<num>'
                self.vocab.add(word)
        self.vocab_size = len(self.vocab)
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(self.vocab)}
        self.eosToken = self.word2idx["<eos>"]

    def tokenize(self, sentence: Union[List[str], str]) -> List[int]:
        if isinstance(sentence, str):
            sentence = sentence.split()
        sentence = [word.lower() if word != 'N' else "<num>" for word in sentence]
        return [self.word2idx[word] for word in sentence]
    
    def detokenize(self, sentence: Union[List[int], int]) -> str:
        if isinstance(sentence, int):
            sentence = [sentence]
        return ' '.join([self.idx2word[idx] if self.idx2word[idx] != "<num>" else 'N' for idx in sentence])
    
    def __call__(self, sentence: Union[List[str], str]) -> List[int]:
        return self.tokenize(sentence)

    def __len__(self) -> int:
        return self.vocab_size