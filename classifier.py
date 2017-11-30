import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

class Classifier(object):
    
    def __init__(self, train_set):
        self.train_set = train_set
        self.train_set_features = None
        self.train_set_labels = None

    def preProcess(self):
        self.train_set_labels = self.train_set['type'].copy()
        colors = self.train_set['color'].copy()
        colors_encoded = pd.get_dummies(colors)
        self.train_set_features = self.train_set.drop(['id', 'type', 'color'], axis=1)
        self.train_set_features['black'] = colors_encoded['black']
        self.train_set_features['blood'] = colors_encoded['blood']
        self.train_set_features['blue'] = colors_encoded['blue']
        self.train_set_features['clear'] = colors_encoded['clear']
        self.train_set_features['green'] = colors_encoded['green']
        self.train_set_features['white'] = colors_encoded['white']

    def classification(self):
        print("Random Forest:")
        clf = RandomForestClassifier(criterion="entropy", n_jobs=-1)
        scores = cross_val_score(clf, self.train_set_features, self.train_set_labels, cv=10, scoring="accuracy")
        print(scores)

        print("KNN")
        clf = KNeighborsClassifier(n_neighbors=5)
        scores = cross_val_score(clf, self.train_set_features, self.train_set_labels, cv=10, scoring="accuracy")
        print(scores)

    def execute(self):
        self.preProcess()
        self.classification()