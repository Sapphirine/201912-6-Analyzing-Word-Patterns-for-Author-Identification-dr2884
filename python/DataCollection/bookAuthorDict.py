import numpy as np
import json
import operator
import pickle
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

BOOK_DATA_DIR_PATH = '/Users/david/ColumbiaCS/bda/project/final-website/bookData/rawData'
BOOK_DATA_PCA_DIR_PATH = '/Users/david/ColumbiaCS/bda/project/final-website/bookData/pcaData'
BOOK_DATA_PLOT_DIR_PATH = '/Users/david/ColumbiaCS/bda/project/final-website/bookData/plotData'
INDEX_JSON_PATH = '/Users/david/ColumbiaCS/bda/project/final-website/bookData/index.json'
FILTERED_INDEX_JSON_PATH = '/Users/david/ColumbiaCS/bda/project/final-website/bookData/filtered_index.json'
TRAINING_INDEX_JSON_PATH = '/Users/david/ColumbiaCS/bda/project/final-website/bookData/training_index.json'
VALIDATION_INDEX_JSON_PATH = '/Users/david/ColumbiaCS/bda/project/final-website/bookData/validation_index.json'
TESTING_INDEX_JSON_PATH = '/Users/david/ColumbiaCS/bda/project/final-website/bookData/testing_index.json'
SCALER_PATH = '/Users/david/ColumbiaCS/bda/project/final-website/bookData/scaler.pkl'
PCA_PATH = '/Users/david/ColumbiaCS/bda/project/final-website/bookData/pca.pkl'

X_TRAIN_PATH = '/Users/david/ColumbiaCS/bda/project/final-website/bookData/dist/X_train'
Y_TRAIN_PATH = '/Users/david/ColumbiaCS/bda/project/final-website/bookData/dist/y_train'
X_VALID_PATH = '/Users/david/ColumbiaCS/bda/project/final-website/bookData/dist/X_valid'
Y_VALID_PATH = '/Users/david/ColumbiaCS/bda/project/final-website/bookData/dist/y_valid'
X_TEST_PATH = '/Users/david/ColumbiaCS/bda/project/final-website/bookData/dist/X_test'
Y_TEST_PATH = '/Users/david/ColumbiaCS/bda/project/final-website/bookData/dist/y_test'




def loadAuthorDict():
    with open(FILTERED_INDEX_JSON_PATH, 'r') as fp:
        return json.load(fp)

def saveAuthorDict(authorDict, name):
    with open(name, 'w') as fp:
        json.dump(authorDict, fp)


def getSortedAuthorCounts(authorDict):
    sortedAuthorCounts = sorted({author: len(lst) for author, lst in authorDict.items()}.items(),
                                key=operator.itemgetter(1), reverse=True)
    N = sum([c for a, c in sortedAuthorCounts])
    trainDict = {}
    validDict = {}
    testDict = {}
    currCount = 0
    for a, c in sortedAuthorCounts:
        if currCount < 0.7 * N:
            trainDict[a] = authorDict[a]
        elif currCount < 0.9 * N:
            validDict[a] = authorDict[a]
        else:
            testDict[a] = authorDict[a]
        currCount += c
    return trainDict, validDict, testDict


def createTrainValidTestDictionaries():
    authorDict = loadAuthorDict()
    trainAuthorDict, validAuthorDict, testAuthorDict = getSortedAuthorCounts(authorDict)
    print("Number of Training Items:", sum([len(lst) for a, lst in trainAuthorDict.items()]))
    print("Number of Validation Items:", sum([len(lst) for a, lst in validAuthorDict.items()]))
    print("Number of Testing Items:", sum([len(lst) for a, lst in testAuthorDict.items()]))
    saveAuthorDict(trainAuthorDict, TRAINING_INDEX_JSON_PATH)
    saveAuthorDict(validAuthorDict, VALIDATION_INDEX_JSON_PATH)
    saveAuthorDict(testAuthorDict, TESTING_INDEX_JSON_PATH)
    return trainAuthorDict, validAuthorDict, testAuthorDict

def getIndexToAuthorList(authorDict):
    return {metadata['index']: author for author, metadataList in authorDict.items() for metadata in metadataList}

def getIndexList(authorDict):
    return [metadata['index'] for author, metadataList in authorDict.items() for metadata in metadataList ]

def getDataList(indexList):
    return [np.nan_to_num(np.load('{}/v{:06}.npy'.format(BOOK_DATA_DIR_PATH, index))) for index in indexList]

def createScaler(trainDataList):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(trainDataList)
    with open(SCALER_PATH, 'wb') as fp:
        pickle.dump(scaler, fp)
    return scaler

def createPCA(scaledTrainList, componentCount = 20):
    pca = PCA(n_components=componentCount)
    pca.fit(scaledTrainList)
    with open(PCA_PATH, 'wb') as fp:
        pickle.dump(pca, fp)
    return pca

def saveData(indexList, dataList, dataPath, prefix):
    for idx, data in zip(indexList, dataList):
        filepath = '{}/{}{:06}'.format(dataPath, prefix, idx)
        np.save(filepath, data)

def getSameAndDiffPairs(reverseDict):
    samePairs = []
    diffPairs = []
    allIndexAuthors = list(reverseDict.items())
    for i in range(len(allIndexAuthors)):
        idxI, authorI = allIndexAuthors[i]
        for j in range(i, len(allIndexAuthors)):
            idxJ, authorJ = allIndexAuthors[j]
            if authorI == authorJ:
                samePairs.append((idxI, idxJ))
            else:
                diffPairs.append((idxI, idxJ))
    return samePairs, diffPairs

def createSameAndDiffPairings(reverseDict):
    samePair, diffPair = getSameAndDiffPairs(reverseDict)
    print('Same/Diff Pair Counts:', len(samePair), len(diffPair))
    return samePair, diffPair


def collectDiffVectors(normL, samePair, sameSize, diffPair, diffSize, indexList, pcaData):
    X_same = [pcaData[indexList.index(idxI)] - pcaData[indexList.index(idxJ)] for idxI, idxJ in samePair[:sameSize]]
    X_diff = [pcaData[indexList.index(idxI)] - pcaData[indexList.index(idxJ)] for idxI, idxJ in diffPair[:diffSize]]
    X_same = [abs(vec)**normL for vec in X_same]
    X_diff = [abs(vec) ** normL for vec in X_diff]

    random.shuffle(X_same)
    random.shuffle(X_diff)

    X = np.concatenate((X_same, X_diff))
    y = np.array([0] * sameSize+ [1] * diffSize)

    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]

    return X, y


def transformAndSaveData(trainAuthorDict, validAuthorDict, testAuthorDict):
    trainReverseDict = getIndexToAuthorList(trainAuthorDict)
    validReverseDict = getIndexToAuthorList(validAuthorDict)
    testReverseDict = getIndexToAuthorList(testAuthorDict)

    trainIndexList = getIndexList(trainAuthorDict)
    validIndexList = getIndexList(validAuthorDict)
    testIndexList = getIndexList(testAuthorDict)

    trainDataList = getDataList(trainIndexList)
    validDataList = getDataList(validIndexList)
    testDataList = getDataList(testIndexList)

    scaler = createScaler(trainDataList)

    scaledTrainList = scaler.transform(trainDataList)
    scaledValidList = scaler.transform(validDataList)
    scaledTestList = scaler.transform(testDataList)

    pca = createPCA(scaledTrainList, componentCount=20)

    pcaTrain = pca.transform(scaledTrainList)
    pcaValid = pca.transform(scaledValidList)
    pcaTest = pca.transform(scaledTestList)

    saveData(trainIndexList, pcaTrain, BOOK_DATA_PCA_DIR_PATH, 'pca')
    saveData(validIndexList, pcaValid, BOOK_DATA_PCA_DIR_PATH, 'pca')
    saveData(testIndexList, pcaTest, BOOK_DATA_PCA_DIR_PATH, 'pca')

    plotTrain = pcaTrain[:,:2]
    plotValid = pcaValid[:, :2]
    plotTest = pcaTest[:, :2]

    saveData(trainIndexList, plotTrain, BOOK_DATA_PLOT_DIR_PATH, 'plt')
    saveData(validIndexList, plotValid, BOOK_DATA_PLOT_DIR_PATH, 'plt')
    saveData(testIndexList, plotTest, BOOK_DATA_PLOT_DIR_PATH, 'plt')

    trainSamePair, trainDiffPair = createSameAndDiffPairings(trainReverseDict)
    validSamePair, validDiffPair = createSameAndDiffPairings(validReverseDict)
    testSamePair, testDiffPair = createSameAndDiffPairings(testReverseDict)

    X_train, y_train = collectDiffVectors(2, trainSamePair, 100000, trainDiffPair, 100000, trainIndexList, pcaTrain)
    X_valid, y_valid = collectDiffVectors(2, validSamePair, 10000, validDiffPair, 10000, validIndexList, pcaValid)
    X_test, y_test = collectDiffVectors(2, testSamePair, 1000, testDiffPair, 1000, testIndexList, pcaTest)

    np.save(X_TRAIN_PATH, X_train)
    np.save(Y_TRAIN_PATH, y_train)
    np.save(X_VALID_PATH, X_valid)
    np.save(Y_VALID_PATH, y_valid)
    np.save(X_TEST_PATH, X_test)
    np.save(X_TEST_PATH, y_test)

    return X_train, y_train, X_valid, y_valid, X_test, y_test