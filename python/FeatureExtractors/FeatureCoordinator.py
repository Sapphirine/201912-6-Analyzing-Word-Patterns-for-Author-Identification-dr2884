from joblib import dump, load

def savePCA(filename, pca):
    dump(pca, filename)

def loadPCA(filename):
    return load(filename)

