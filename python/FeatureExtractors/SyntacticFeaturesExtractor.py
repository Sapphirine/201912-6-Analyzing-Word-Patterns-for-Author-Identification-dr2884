import numpy as np
import string
import nltk
from .FeatureExtractor import FeatureExtractor


class PunctuationFrequncyExtractor(FeatureExtractor):
    def __init__(self):
        pass

    def extract(self, text):
        freq = np.array([text.count(p) for p in string.punctuation])
        rel = freq / sum(freq)
        return np.concatenate((freq, rel))

class FunctionWordFrequencyExtractor(FeatureExtractor):
    def __init__(self):
        self.functionWordList = ['a', 'about' 'above', 'after', 'all', 'although', 'am', 'among', 'an', 'and',
                                 'another', 'any', 'anybody', 'anyone', 'anything', 'are', 'around', 'as', 'at',
                                 'be', 'because', 'before', 'behind', 'below', 'beside', 'between', 'both', 'but',
                                 'by', 'can', 'cos', 'do', 'down', 'each', 'either', 'enough', 'every', 'everybody',
                                 'everyone', 'everything', 'few', 'following', 'for', 'from', 'have', 'he', 'her',
                                 'him', 'i', 'if', 'in', 'including', 'inside', 'into', 'is', 'it', 'its', 'latter',
                                 'less', 'like', 'little', 'lots', 'many', 'me', 'more', 'most', 'much', 'must',
                                 'my', 'near', 'need', 'neither', 'no', 'nobdy', 'none', 'nor', 'nothing', 'of',
                                 'off', 'on', 'once', 'one', 'onto', 'opposite', 'or', 'our', 'outside', 'over',
                                 'own', 'past', 'per', 'plenty', 'plus', 'regarding', 'same', 'several', 'she',
                                 'should', 'since', 'so', 'some', 'somebody', 'someone', 'something', 'such',
                                 'than', 'that', 'the', 'their', 'them', 'these', 'they', 'this', 'those', 'though',
                                 'till', 'to', 'toward', 'towards', 'under', 'unless', 'unlike', 'until', 'up',
                                 'upon', 'us', 'used', 'via', 'we', 'what', 'whatever', 'when', 'where', 'whether',
                                 'which', 'while', 'who', 'whoever', 'whom', 'whose', 'will', 'with', 'within',
                                 'without', 'worth', 'would', 'yes', 'you', 'your']

    def extract(self, text):
        wordList = [token.lower() for token in nltk.tokenize.word_tokenize(text) \
                    if token not in string.punctuation]
        freq = np.array([wordList.count(fw) for fw in self.functionWordList])
        rel = freq / len(wordList)
        fwRel = freq / sum(freq)
        return np.concatenate((freq, rel, fwRel))

class POSMonogramFrequencyExtractor(FeatureExtractor):
    def __init__(self):
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.posTokenList = ['#', '$', "''", '(', ')', ',', '.', ':', 'CC', 'CD', 'DT', 'EX',
                             'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS',
                             'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP',
                             'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                             'WDT', 'WP', 'WP$', 'WRB', '``']
        self.posIndexMap = {k: v for v, k in enumerate(self.posTokenList)}

    def extract(self, text):
        text = text.strip().lower()
        posList = [pair[1] for pair in nltk.pos_tag(nltk.word_tokenize(text))]
        freq = np.array([posList.count(pos) for pos in self.posTokenList])
        rel = freq / sum(freq)
        return np.concatenate((freq, rel))

class POSBigramFrequencyExtractor(FeatureExtractor):
    def __init__(self):
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.posTokenList = ['#', '$', "''", '(', ')', ',', '.', ':', 'CC', 'CD', 'DT', 'EX',
                             'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS',
                             'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP',
                             'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                             'WDT', 'WP', 'WP$', 'WRB', '``']
        self.posIndexMap = {k: v for v, k in enumerate(self.posTokenList)}

    def extract(self, text):
        text = text.strip().lower()
        posList = [pair[1] for pair in nltk.pos_tag(nltk.word_tokenize(text))]
        freq2D = np.zeros(shape=(len(self.posTokenList), len(self.posTokenList)))
        for i in range(len(posList) - 1):
            posU = posList[i]
            posV = posList[i + 1]
            idxU = self.posIndexMap[posU]
            idxV = self.posIndexMap[posV]
            freq2D[idxU][idxV] += 1

        rel2D = np.zeros(shape=(len(self.posTokenList), len(self.posTokenList)))

        for i in range(len(freq2D)):
            rowSum = sum(freq2D[i])
            if rowSum == 0:
                continue
            else:
                rel2D[i] = freq2D[i] / rowSum

        freq = freq2D.ravel()
        rel = freq2D.ravel()
        return np.concatenate((freq, rel))

class SyntacticFeaturesExtractor(FeatureExtractor):
    def __init__(self):
        self.subFeatureExtractors = [
            PunctuationFrequncyExtractor(),
            FunctionWordFrequencyExtractor(),
            POSMonogramFrequencyExtractor(),
            POSBigramFrequencyExtractor()
        ]

    def extract(self, text):
        return np.concatenate([fe.extract(text) for fe in self.subFeatureExtractors])