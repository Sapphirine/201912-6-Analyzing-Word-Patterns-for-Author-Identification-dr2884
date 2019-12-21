from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
from gutenberg.query import get_metadata, list_supported_metadatas
from gutenberg.acquire import get_metadata_cache
from gutenberg._domain_model.exceptions import UnknownDownloadUriException
import json
import tqdm
import os
from ..FeatureExtractors import CompleteFeaturesExtractor

INDEX_JSON_PATH = '/Users/david/ColumbiaCS/bda/project/data/index.json'
VECTOR_SAVE_DIR_PATH = '/Users/david/ColumbiaCS/bda/project/data'

def loadAuthorDict():
    with open(INDEX_JSON_PATH, 'r') as fp:
        return json.load(fp)

def collectDataVectors(authorDict, featureExtractor):
    pairList = [(k, metadata) for (k,v) in authorDict.items() for metadata in v]
    for author, metadataDict in tqdm(pairList[40596:][9200:]):
        index = metadataDict['index']
        filename = '{}/v{:06}'.format(VECTOR_SAVE_DIR_PATH, index)
        if os.path.isfile(filename+'.npy'):
            continue

        try:
            text = strip_headers(load_etext(index)).strip()
            vector = featureExtractor.extract(text)
            np.save(filename, vector)
        except UnknownDownloadUriException:
            noDownload.append(index)
            pass
        except ZeroDivisionError:
            noDownload.append(index)
            pass

authorDict = loadAuthorDict()
noDownload = []
collectDataVectors(authorDict, CompleteFeaturesExtractor())