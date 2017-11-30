from analyzer import Analyzer
from classifier import Classifier
import pandas as pd

train_set = pd.read_csv('train.csv', header=0, index_col=None)

# analyzer = Analyzer(train_set)
# analyzer.process()
classifier = Classifier(train_set)
classifier.execute()