import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

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

        colors = dataset['color'].copy()
        colors_encoded = pd.get_dummies(colors)
        dataset_features = dataset.drop(['id', 'color'], axis=1)
        dataset_features['black'] = colors_encoded['black']
        dataset_features['blood'] = colors_encoded['blood']
        dataset_features['blue'] = colors_encoded['blue']
        dataset_features['clear'] = colors_encoded['clear']
        dataset_features['green'] = colors_encoded['green']
        dataset_features['white'] = colors_encoded['white']

        return dataset_features

    def classification(self):
        # print("Random Forest:")
        # clf = RandomForestClassifier(criterion="entropy", n_jobs=-1)
        # scores = cross_val_score(clf, self.train_set_features, self.train_set_labels, cv=10, scoring="accuracy")
        # print(np.mean(scores))

        # print("KNN")
        # clf = KNeighborsClassifier(n_neighbors=5)
        # scores = cross_val_score(clf, self.train_set_features, self.train_set_labels, cv=10, scoring="accuracy")
        # print(np.mean(scores))

        # print("MLP")
        # clf = MLPClassifier(hidden_layer_sizes=(7,), max_iter=1200)
        # scores = cross_val_score(clf, self.train_set_features, self.train_set_labels, cv=10, scoring='accuracy')
        # print(np.mean(scores))

        clf = MLPClassifier(hidden_layer_sizes=(7,), max_iter=1200)
        clf.fit(self.train_set_features, self.train_set_labels)
        predict = clf.predict(self.test_set_features)

        return predict


    def execute(self):
        self.preProcess()
        return self.classification()