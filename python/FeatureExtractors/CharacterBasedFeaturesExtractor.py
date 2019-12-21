import numpy as np
from .FeatureExtractor import FeatureExtractor

class NumberOfCharactersExtractor(FeatureExtractor):
    def __init__(self):
        pass

    def extract(self, text):
        return np.array([len(text)])

class NumberOfAlphabeticCharactersExtractor(FeatureExtractor):
    def __init__(self):
        pass

    def extract(self, text):
        numAlpha = len([c for c in text if c.isalpha()])
        N = len(text)
        return np.array([numAlpha, numAlpha / N])

class NumberOfUppercaseCharactersExtractor(FeatureExtractor):
    def __init__(self):
        pass

    def extract(self, text):
        numUpper = len([c for c in text if c.isupper()])
        N = len(text)
        return np.array([numUpper, numUpper / N])

class NumberOfDigitCharactersExtractor(FeatureExtractor):
    def __init__(self):
        pass

    def extract(self, text):
        numDigit = len([c for c in text if c.isdigit()])
        N = len(text)
        return np.array([numDigit, numDigit / N])

# Counts whitespace characters: [‘ ‘, ‘\t’, ‘\n’, ‘\v’, ‘\f’, ‘\r’]
class NumberOfWhiteSpaceCharactersExtractor(FeatureExtractor):
    def __init__(self):
        pass

    def extract(self, text):
        numWhitespace = len([c for c in text if c.isspace()])
        N = len(text)
        return np.array([numWhitespace, numWhitespace / N])

class NumberOfTabCharactersExtractor(FeatureExtractor):
    def __init__(self):
        pass

    def extract(self, text):
        numTabs = text.count('\t')
        N = len(text)
        return np.array([numTabs, numTabs / N])

class NumberOfSpacesExtractor(FeatureExtractor):
    def __init__(self):
        pass

    def extract(self, text):
        numSpace = text.count(' ')
        N = len(text)
        return np.array([numSpace, numSpace / N])

class NumberOfNewlinesExtractor(FeatureExtractor):
    def __init__(self):
        pass

    def extract(self, text):
        numNewline = text.count('\n')
        N = len(text)
        return np.array([numNewline, numNewline / N])

class FrequencyOfAlphabetLetter(FeatureExtractor):
    def __init__(self):
        self.alphabetArray = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', \
                              'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', \
                              's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    def extract(self, text):
        totalCharLength = len(text)
        freq = np.array([text.count(c) for c in self.alphabetArray])
        rel = freq / totalCharLength
        return np.concatenate((freq, rel))


class FrequencyOfSpecialCharacters(FeatureExtractor):
    def __init__(self):
        self.specialCharacterArray = ['~', '@', '#', '$', '%', '^', '&', '*', '-', \
                                      '_', '=', '+', '>', '<', '[', ']', '{', '}', \
                                      '/', '\\', '|']

    def extract(self, text):
        totalCharLength = len(text)
        freq = np.array([text.count(c) for c in self.specialCharacterArray])
        rel = freq / totalCharLength
        return np.concatenate((freq, rel))

class CharacterBasedFeaturesExtractor(FeatureExtractor):
    def __init__(self):
        self.subFeatureExtractors = [
            NumberOfCharactersExtractor(),
            NumberOfAlphabeticCharactersExtractor(),
            NumberOfUppercaseCharactersExtractor(),
            NumberOfDigitCharactersExtractor(),
            NumberOfWhiteSpaceCharactersExtractor(),
            NumberOfTabCharactersExtractor(),
            NumberOfSpacesExtractor(),
            NumberOfNewlinesExtractor(),
            FrequencyOfAlphabetLetter(),
            FrequencyOfSpecialCharacters()
        ]

    def extract(self, text):
        return np.concatenate([fe.extract(text) for fe in self.subFeatureExtractors])