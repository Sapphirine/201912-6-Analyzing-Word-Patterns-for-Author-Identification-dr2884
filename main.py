# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START gae_python37_render_template]
import datetime

from flask import Flask, render_template, request
from python.FeatureExtractors import CompleteFeaturesExtractor
from python.Classifiers.EnsembleClassifier import EnsembleSVC, loadEnsembleSVC
from python.DataCollection.bookAuthorDict import loadAuthorDict, getDataList

import random
import numpy as np
# import json

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

app = Flask(__name__)
ensembleSVC = loadEnsembleSVC()

# prevent cached responses
if app.config["DEBUG"]:
    @app.after_request
    def after_request(response):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
        response.headers["Expires"] = 0
        response.headers["Pragma"] = "no-cache"
        return response

@app.route('/', methods=['GET', 'POST'])
def root():
    # For the sake of example, use static information to inflate the template.
    # This will be replaced with real information in later steps.
    dummy_times = [datetime.datetime(2018, 1, 1, 10, 0, 0),
                   datetime.datetime(2018, 1, 2, 10, 30, 0),
                   datetime.datetime(random.randint(1,2018), 1, 3, 11, 0, 0),
                   ]

    if request.method == "POST":
        text1 = request.form['text1']
        text2 = request.form['text2']

        fe = CompleteFeaturesExtractor(True)
        print("Extracting Features - 1")
        features1 = fe.extract(text1)
        print("Extracting Features - 2")
        features2 = fe.extract(text2)
        print("Calculating Distnace")
        distance = np.linalg.norm(features1 - features2)

        print("Calculating Prediction")
        # prediction = 'True' if ensembleSVC.predict((features1-features2)**2, 1) == 0 else 'False'
        prediction = 'True' if text2[1] == 'h' else 'False'

        print("Returning Data")
        data = [dummy_times, str(features1), str(features2), distance, prediction, text1, text2]
    else:
        data = [dummy_times, '', '', '', '', 'Enter Text Here...', 'Enter Text Here...']

    return render_template('index.html', data=data)

@app.route('/compare', methods=['GET', 'POST'])
def compare():
    authorDict = loadAuthorDict()
    authorList = sorted(list(authorDict.keys()))

    if request.method == "POST":
        author1 = request.form['author1']
        author2 = request.form['author2']
    else:
        author1 = 'Conrad, Joseph'
        author2 = 'Darwin, Charles'

    indexList1 = [metadata['index'] for metadata in authorDict[author1]]
    indexList2 = [metadata['index'] for metadata in authorDict[author2]]

    titleList1 = [metadata['title'] for metadata in authorDict[author1]]
    titleList2 = [metadata['title'] for metadata in authorDict[author2]]

    dataList1 = getDataList(indexList1)
    dataList2 = getDataList(indexList2)

    steps = [('scaler', MinMaxScaler(feature_range=(0, 1))), ('pca', PCA(n_components=2))]
    pipeline = Pipeline(steps)
    pipeline.fit(dataList1+dataList2)

    pointList1 = pipeline.transform(dataList1).tolist()
    pointList2 = pipeline.transform(dataList2).tolist()

    tupList = list(zip([author1 for idx in indexList1]+ [author2 for idx in indexList2], titleList1+titleList2, pointList1+pointList2))
    data = [{'author': tup[0], 'title': tup[1], 'x': tup[2][0], 'y': tup[2][1]} for tup in tupList]

    data = [authorList,
            author1, len(indexList1), titleList1,
            author2, len(indexList2), titleList2,
            data]

    return render_template('compare.html', data=data)

# @app.route('/', methods=['GET', 'POST'])
# def submitForm():
#     text = request.form('text')
#     processed_text = text.upper()
#     print(processed_text)
#     return render_template('index.html', data=data)

if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    # Flask's development server will automatically serve static files in
    # the "static" directory. See:
    # http://flask.pocoo.org/docs/1.0/quickstart/#static-files. Once deployed,
    # App Engine itself will serve those files as configured in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
# [START gae_python37_render_template]
