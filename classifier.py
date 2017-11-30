import numpy as np
import pandas as pd

class Classifier(object):
    
    def __init__(self, train_set):
        self.train_set = train_set

    def preProcess(self):
        colors = self.train_set['color'].copy()
        colors_encoded = pd.get_dummies(colors)
        self.train_set_features = self.train_set.drop(['id', 'type', 'color'], axis=1)
        self.train_set_features = np.concatenate((self.train_set_features, colors_encoded), axis=1)
        print(self.train_set_features)
