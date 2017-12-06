from analyzer import Analyzer
from classifier import Classifier
import pandas as pd

train_set = pd.read_csv('train.csv', header=0, index_col=None)
test_set = pd.read_csv('test.csv', header=0, index_col=None)

result = pd.DataFrame(test_set['id'].copy())

analyzer = Analyzer(train_set)
analyzer.process()
classifier = Classifier(train_set, test_set)
labels, predict = classifier.testModel()
analyzer.evaluate(labels, predict)

predict = classifier.execute()  # using test set
result['type'] = predict
result.to_csv("result.csv", sep=',', index=False)