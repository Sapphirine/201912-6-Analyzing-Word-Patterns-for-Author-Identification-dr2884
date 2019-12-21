import numpy as np
import pickle
from multiprocessing import Pool

from sklearn.svm import SVC, SVR


ENSEMBLE_FILE_PATH = '/Users/david/ColumbiaCS/bda/project/final-website/python/Classifiers/ensemble.pkl'


class EnsembleSVC(object):
    def __init__(self, numSvr, pcaCount=None, kernel='rbf', C=10, gamma=0.1, degree=2):
        self.numSvr = numSvr
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.svrList = [SVR(kernel=kernel, C=C, degree=degree, gamma=gamma) for x in range(numSvr)]
        self.svc = SVC(kernel='poly', C=C, degree=degree, gamma=gamma)
        self.pca = PCA(n_components=pcaCount) if pcaCount else None

    def trainSVR(self, svr, X, y):
        svr.fit(X, y)
        return svr

    def predictSVR(self, svr, X):
        return svr.predict(X)

    def fit(self, X_train, y_train, n_jobs=8):
        p = Pool(n_jobs)
        dataSplitX = np.array_split(X_train, self.numSvr + 1)
        dataSplitY = np.array_split(y_train, self.numSvr + 1)
        trainingListX = dataSplitX[:-1]
        trainingListY = dataSplitY[:-1]
        validationX = dataSplitX[-1]
        validationY = dataSplitY[-1]
        self.svrList = p.starmap(self.trainSVR, zip(self.svrList, trainingListX, trainingListY))

        predList = p.starmap(self.predictSVR, zip(self.svrList, [validationX for idx in range(self.numSvr)]))
        svrFeatures = np.stack(predList, axis=1)
        pcaFeatures = self.pca.fit_transform(svrFeatures) if self.pca else svrFeatures
        self.svc.fit(pcaFeatures, validationY)

    def predict(self, X, n_jobs=8):
        p = Pool(n_jobs)
        predList = p.starmap(self.predictSVR, zip(self.svrList, [X for idx in range(self.numSvr)]))
        svrFeatures = np.stack(predList, axis=1)
        pcaFeatures = self.pca.transform(svrFeatures) if self.pca else svrFeatures
        return self.svc.predict(pcaFeatures)


def loadEnsembleSVC():
    with open(ENSEMBLE_FILE_PATH, 'rb') as file:
        return pickle.load(file)