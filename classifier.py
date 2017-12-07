import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import seaborn as sns

class Classifier(object):
    
    def __init__(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set
        self.train_set_features = None
        self.train_set_labels = None
        self.test_set_features = None
        self.test_set_labels = None

    def preProcess(self):
        self.train_set_features = self.preProcessSet('train')
        self.test_set_features = self.preProcessSet('test')

    def preProcessSet(self, type):
        dataset = None
        dataset_features = None
        if type == 'train':
            dataset = self.train_set
            self.train_set_labels = self.train_set['type'].copy()
            dataset = dataset.drop('type', axis=1)
        else:
            dataset = self.test_set
        dataset_features = dataset.drop(['id', 'color'], axis=1)
        
        ## Encode color attribute. Commented because we don't use this attribute.

        # colors = dataset['color'].copy()
        # colors_encoded = pd.get_dummies(colors)
        # dataset_features = dataset.drop(['id', 'color'], axis=1)
        # dataset_features['black'] = colors_encoded['black']
        # dataset_features['blood'] = colors_encoded['blood']
        # dataset_features['blue'] = colors_encoded['blue']
        # dataset_features['clear'] = colors_encoded['clear']
        # dataset_features['green'] = colors_encoded['green']
        # dataset_features['white'] = colors_encoded['white']

        return dataset_features

    def findBestParameters(self):
        params = {'solver': ['adam', 'sgd'], 'activation': ['relu', 'tanh'], 'max_iter': [800, 1200, 1500], 'alpha': [0.0001, 0.01, 0.1], 'hidden_layer_sizes': np.arange(4, 10)}
        clf_grid = GridSearchCV(estimator=MLPClassifier(), param_grid=params, scoring='accuracy', cv=5) 
        clf_grid.fit(self.train_set_features, self.train_set_labels)

        print(clf_grid.best_score_)
        print(clf_grid.best_params_)

    def testModel(self):
        train_set, test_set = train_test_split(self.train_set, test_size = 0.1, random_state = 42)
        train_labels = train_set['type'].copy()
        train_features = train_set.drop(['type', 'id', 'color'], axis=1)
        test_labels = test_set['type'].copy()
        test_features = test_set.drop(['type', 'id', 'color'], axis=1)
        
        clf = MLPClassifier(solver='adam', hidden_layer_sizes=(9,), max_iter=800, alpha=0.01)
        clf.fit(train_features, train_labels)
        predict = clf.predict(test_features)

        return test_labels, predict

    def compareModels(self):
        print("Comparing models...")
        print("Random Forest:")
        clf = RandomForestClassifier(criterion="entropy", n_jobs=-1)
        scores = cross_val_score(clf, self.train_set_features, self.train_set_labels, cv=10, scoring="accuracy")
        print(np.mean(scores))

        print("KNN")
        clf = KNeighborsClassifier(n_neighbors=5)
        scores = cross_val_score(clf, self.train_set_features, self.train_set_labels, cv=10, scoring="accuracy")
        print(np.mean(scores))

        print("MLP")
        clf = MLPClassifier(solver='adam', hidden_layer_sizes=(7,), max_iter=1000, alpha=0.0001)
        scores = cross_val_score(clf, self.train_set_features, self.train_set_labels, cv=10, scoring='accuracy')
        print(np.mean(scores))

    def classification(self):
        clf = MLPClassifier(solver='adam', hidden_layer_sizes=(9,), max_iter=800, alpha=0.01)
        clf.fit(self.train_set_features, self.train_set_labels)
        predict = clf.predict(self.test_set_features)

        return predict


    def execute(self):
        self.preProcess()
        self.compareModels()
        return self.classification()