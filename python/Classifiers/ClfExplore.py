from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, explained_variance_score


def trainSVC(X_train, y_train, X_valid, y_valid):
    param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [1, 2, 3]},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    gs = GridSearchCV(estimator=SVC(), param_grid=param_grid, verbose=2, n_jobs=1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_


def trainSVR(X_train, y_train, X_valid, y_valid):
    param_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'kernel': ['poly'], 'degree': [1, 2, 3]},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    ]

    gs = GridSearchCV(estimator=SVR(), param_grid=param_grid, verbose=2, n_jobs=1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_


def trainMLP(X_train, y_train, X_valid, y_valid):
    param_grid = [
        {'activation': ['identity', 'logistic', 'tanh', 'relu'],
         'batch_size': [200, 1000], 'hidden_layer_sizes': [(100,), (100, 50,), (100, 50, 10,)]},
    ]

    gs = GridSearchCV(estimator=MLPClassifier(), param_grid=param_grid, verbose=2, n_jobs=1)
    gs.fit(X_train, y_train)
    return gs.best_estimator_

def evaluateClassifier(clf, X_valid, y_valid):
    y_pred = clf.predict(X_valid)
    print(confusion_matrix(y_valid,y_pred))
    print(classification_report(y_valid,y_pred))
    print("Accuracy:", sum(y_pred == y_valid) / len(y_valid))