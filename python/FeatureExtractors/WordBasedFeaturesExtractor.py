import numpy as np
import string
import math
import nltk
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from collections import Counter

from .FeatureExtractor import FeatureExtractor


class NumberOfWordsExtractor(FeatureExtractor):
    def __init__(self):
        pass

    def extract(self, text):
        wordList = [token for token in nltk.tokenize.word_tokenize(text) if token not in string.punctuation]
        return np.array([len(wordList)])

class NumberOfCharactersInWordsExtractor(FeatureExtractor):
    def __init__(self):
        pass

    def extract(self, text):
        wordList = [token for token in nltk.tokenize.word_tokenize(text) if token not in string.punctuation]
        numCharsInWords = sum([len(w) for w in wordList])
        N = len(text)
        return np.array([numCharsInWords, numCharsInWords / N])

class NumberOfShortWordsExtractor(FeatureExtractor):
    def __init__(self):
        pass

    def extract(self, text):
        wordList = [token for token in nltk.tokenize.word_tokenize(text) if token not in string.punctuation]
        numShortWords = len([w for w in wordList if len(w) <= 3])
        N = len(wordList)
        return np.array([numShortWords, numShortWords / N])

class AverageWordLengthExtractor(FeatureExtractor):
    def __init__(self):
        pass

    def extract(self, text):
        wordList = [token for token in nltk.tokenize.word_tokenize(text) if token not in string.punctuation]
        return np.array([sum([len(w) for w in wordList]) / len(wordList)])

class AverageSentenceLengthByCharacterExtractor(FeatureExtractor):
    def __init__(self):
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def extract(self, text):
        sentenceList = self.tokenizer.tokenize(text)
        return np.array([sum(len(s) for s in sentenceList) / len(sentenceList)])

class AverageSentenceLengthByWordExtractor(FeatureExtractor):
    def __init__(self):
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def extract(self, text):
        sentenceList = self.tokenizer.tokenize(text)
        totalWordCount = sum(
            len([token for token in nltk.tokenize.word_tokenize(sentence) if token not in string.punctuation]) \
            for sentence in sentenceList)
        return np.array([totalWordCount / len(sentenceList)])

class NumberOfDifferentWordsExtractor(FeatureExtractor):
    def __init__(self):
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def extract(self, text):
        sentenceList = self.tokenizer.tokenize(text)
        wordListList = [[token for token in nltk.tokenize.word_tokenize(sentence) if token not in string.punctuation] \
                        for sentence in sentenceList]
        wordSet = set.union(*map(set, [set(wordList) for wordList in wordListList]))
        N = sum(len(wordList) for wordList in wordListList)
        numDiffWords = len(wordSet)

        return np.array([numDiffWords, numDiffWords / N])

class NumberOfDifferentWordStemsExtractor(FeatureExtractor):
    def __init__(self):
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.stemmer = SnowballStemmer('english')

    def extract(self, text):
        sentenceList = self.tokenizer.tokenize(text)
        wordListList = [[token for token in nltk.tokenize.word_tokenize(sentence) \
                         if token not in string.punctuation] \
                        for sentence in sentenceList]
        wordSet = set.union(*map(set, [set(wordList) for wordList in wordListList]))
        stemSet = set(self.stemmer.stem(w) for w in wordSet)
        numStems = len(stemSet)
        N = sum(len(wordList) for wordList in wordListList)
        return np.array([numStems, numStems / N])

class NumberOfDifferentWordLemmasExtractor(FeatureExtractor):
    def __init__(self):
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(self, treebank_tag):
        # https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def lemmatize(self, posPair):
        wordnetPos = self.get_wordnet_pos(posPair[1])
        if wordnetPos == None:
            return posPair[0]
        else:
            return self.lemmatizer.lemmatize(posPair[0], pos=wordnetPos)

    def extract(self, text):
        sentenceList = self.tokenizer.tokenize(text)
        posPairSetList = [set(pair for pair in nltk.pos_tag(nltk.tokenize.word_tokenize(sentence)) \
                              if pair[0] not in string.punctuation) \
                          for sentence in sentenceList]
        N = sum(len([pair for pair in nltk.tokenize.word_tokenize(sentence) if pair[0] not in string.punctuation]) for
                sentence in sentenceList)
        pairSet = set.union(*map(set, posPairSetList))
        lemmaSet = set(self.lemmatize(p) for p in pairSet)
        numLemmas = len(lemmaSet)
        return np.array([numLemmas, numLemmas / N])

class HapaxLegomenaExtractor(FeatureExtractor):
    def __init__(self):
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def extract(self, text):
        wordList = [token for token in nltk.tokenize.word_tokenize(text) if token not in string.punctuation]
        wordCounts = Counter(wordList)
        onceOccuringWords = [word for (word, count) in wordCounts.items() if count == 1]
        numHL = len(onceOccuringWords)
        N = len(wordList)
        return np.array([numHL, numHL / N])

class HapaxDislegomenaExtractor(FeatureExtractor):
    def __init__(self):
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    def extract(self, text):
        wordList = [token for token in nltk.tokenize.word_tokenize(text) if token not in string.punctuation]
        wordCounts = Counter(wordList)
        twiceOccuringWords = [word for (word, count) in wordCounts.items() if count == 2]
        numHD = 2 * len(twiceOccuringWords)
        N = len(wordList)
        return np.array([numHD, numHD / N])

class YuleKMeasureExtractor(FeatureExtractor):
    # https://www.mitpressjournals.org/doi/pdf/10.1162/COLI_a_00228
    def __init__(self):
        self.C = 10 ** 4

    def extract(self, text):
        wordList = nltk.tokenize.word_tokenize(text)
        wordCounter = Counter(word.lower() for word in wordList)
        frequencySpectrum = Counter(wordCounter.values())
        N = len(wordList)
        leftD = -1 / N
        rightD = sum(numKeys * (count / N) ** 2 for count, numKeys in frequencySpectrum.items())
        if leftD + rightD == 0:
            return np.array([self.C, 0])
        i = 1 / (leftD + rightD)
        k = (1 / i) * self.C
        return np.array([k, i])

class SimpsonDMeasureExtractor(FeatureExtractor):
    # https://www.mitpressjournals.org/doi/pdf/10.1162/COLI_a_00228
    # https://github.com/tsproisl/Linguistic_and_Stylistic_Complexity/blob/master/complexity_measures/vocabulary_richness.py
    def __init__(self):
        pass

    def extract(self, text):
        wordList = nltk.tokenize.word_tokenize(text)
        wordCounter = Counter(word.lower() for word in wordList)
        frequencySpectrum = Counter(wordCounter.values())
        N = len(wordList)
        d = sum(numKeys * (count / N if N != 0 else 0) * ((count - 1) / (N - 1) if N != 1 else 0) for count, numKeys in
                frequencySpectrum.items())
        return np.array([d])

class SichelSMeasureExtractor(FeatureExtractor):
    def __init__(self):
        pass

    def extract(self, text):
        wordList = nltk.tokenize.word_tokenize(text)
        wordCounter = Counter(word.lower() for word in wordList)
        frequencySpectrum = Counter(wordCounter.values())
        V = len(set(wordList))
        return np.array([frequencySpectrum.get(2, 0) / V])

class BrunetWMeasureExtractor(FeatureExtractor):
    def __init__(self):
        self.A = -0.172

    def extract(self, text):
        wordList = nltk.tokenize.word_tokenize(text)
        N = len(wordList)
        V = len(set(wordList))
        return np.array([N ** (V ** -self.A)])

class HonoreRMeasureExtractor(FeatureExtractor):
    def __init__(self):
        pass

    def extract(self, text):
        wordList = nltk.tokenize.word_tokenize(text)
        wordCounter = Counter(word.lower() for word in wordList)
        frequencySpectrum = Counter(wordCounter.values())
        N = len(wordList)
        V = len(set(wordList))
        singletonCount = frequencySpectrum.get(1, 0)
        if singletonCount == V:
            singletonCount -= 1
        return np.array([100 * (math.log(N) / (1 - (singletonCount / V)))])

class WordLengthFrequenciesExtractor(FeatureExtractor):
    def __init__(self):
        self.wordLengthLimit = 20

    def extract(self, text):
        wordList = [token for token in nltk.tokenize.word_tokenize(text) if token not in string.punctuation]
        lengthList = [len(w) for w in wordList]
        lengthCounter = dict(Counter(lengthList))
        N = len(wordList)
        freq = np.array([lengthCounter.get(length, 0) for length in range(self.wordLengthLimit+1)])
        rel = freq/N
        return np.concatenate((freq, rel))

class WordBasedFeaturesExtractor(FeatureExtractor):
    def __init__(self):
        self.subFeatureExtractors = [
            NumberOfWordsExtractor(),
            NumberOfShortWordsExtractor(),
            NumberOfCharactersInWordsExtractor(),
            AverageWordLengthExtractor(),
            AverageSentenceLengthByCharacterExtractor(),
            AverageSentenceLengthByWordExtractor(),
            NumberOfDifferentWordsExtractor(),
            NumberOfDifferentWordStemsExtractor(),
            NumberOfDifferentWordLemmasExtractor(),
            HapaxLegomenaExtractor(),
            HapaxDislegomenaExtractor(),
            YuleKMeasureExtractor(),
            SimpsonDMeasureExtractor(),
            SichelSMeasureExtractor(),
            BrunetWMeasureExtractor(),
            HonoreRMeasureExtractor(),
            WordLengthFrequenciesExtractor()
        ]

    def extract(self, text):
        return np.concatenate([fe.extract(text) for fe in self.subFeatureExtractors])