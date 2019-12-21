import numpy as np
import pickle
from .FeatureExtractor import FeatureExtractor
from .CharacterBasedFeaturesExtractor import CharacterBasedFeaturesExtractor
from .WordBasedFeaturesExtractor import WordBasedFeaturesExtractor
from .SyntacticFeaturesExtractor import SyntacticFeaturesExtractor

SCALER_FILE_PATH = '/Users/david/ColumbiaCS/bda/project/final-website/python/FeatureExtractors/scaler.pkl'
PCA_FILE_PATH = '/Users/david/ColumbiaCS/bda/project/final-website/python/FeatureExtractors/pca.pkl'

class CompleteFeaturesExtractor(FeatureExtractor):
    def __init__(self, clean):
        self.subFeatureExtractors = [
            CharacterBasedFeaturesExtractor(),
            WordBasedFeaturesExtractor(),
            SyntacticFeaturesExtractor()
        ]
        self.clean = clean

        with open(SCALER_FILE_PATH, 'rb') as file:
            self.scaler = pickle.load(file)
        with open(PCA_FILE_PATH, 'rb') as file:
            self.pca = pickle.load(file)

    def extract(self, text):
        rawFeatures = np.nan_to_num(np.concatenate([fe.extract(text) for fe in self.subFeatureExtractors]))
        if self.clean:
            scaledFeatures = self.scaler.transform(rawFeatures.reshape(1, -1))
            pcaFeatures = self.pca.transform(scaledFeatures)
            return pcaFeatures
        return rawFeatures

