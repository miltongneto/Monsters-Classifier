import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
import seaborn as sns

class Analyzer(object):
    
    def __init__(self, train_set):
        self.train_set = train_set
        self.classes = ['Ghoul', 'Goblin', 'Ghost']

    def histogram(self):
        labels = self.train_set.drop('id', axis=1)
        labels.hist()
        plt.show()

    def barCategoricalData(self):
        colors_data = self.train_set['color'].copy()
        count = Counter(colors_data)
        
        clear = count['clear']
        green = count['green']
        black = count['black']
        white = count['white']
        blue = count['blue'] 
        blood = count['blood']

        bar_heights = (clear, green, black, white, blue, blood)
        x = ('clear', 'green', 'black', 'white', 'blue', 'blood')

        fig, ax = plt.subplots()
        width = 0.3

        ax.bar(x, bar_heights, width)
        plt.show()

    def classBalance(self): 
        labels = self.train_set['type'].copy()
        plt.pie(labels.value_counts(), labels=['Ghoul', 'Globin', 'Ghost'],autopct='%1.1f%%', shadow=True)
        plt.title("Distribuição das classes")
        plt.show()
    
    def distributionAttributes(self):
        # generate color distribution for each class
        sns.factorplot("type", col="color", col_wrap=4, data=self.train_set, kind="count", size=2.4, aspect=.8)
        
        # generate numerical attribute distribution for each class
        fig = plt.figure(figsize=(8, 8))
        idx = 1
        numeric_columns = ['bone_length', 'rotting_flesh', 'hair_length', 'has_soul']
        for f in numeric_columns:
            ax = fig.add_subplot(2, len(numeric_columns) / 2, idx)
            idx += 1
            sns.boxplot(x='type', y=f, data=self.train_set, palette='muted', ax=ax)

    def verifyMissValues(self):
        print("Missing data:")
        columnsWithNaN = self.train_set.isnull().any()
        print(columnsWithNaN)
    
    def pairs(self):
        labels = self.train_set['type'].copy()
        features = self.train_set.drop(['id','color'], axis=1)
        pairs = sns.pairplot(features, hue='type')
        pairs.savefig("pares_features.png")
        plt.show()

    def evaluate(self, labels, predict):
        score = accuracy_score(labels, predict)
        print("Accuracy: ", score)
        self.showConfusionMatrix(labels, predict)
        #self.showROCCurve(labels, predict)

    def showConfusionMatrix(self, labels, predict):
        cm = confusion_matrix(labels,predict,labels=self.classes)
        sns.heatmap(cm,annot=True, xticklabels=self.classes, yticklabels=self.classes, cmap="Blues").set(xlabel = "Predicted Class", ylabel = "True Class", title = "Confusion Matrix")
        plt.show()

    def showROCCurve(self, labels, predict):
        # Generate ROC Curve, incomplete. Not used
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        teste = np.array(labels)
        teste2 = np.array(predict)
        fpr, tpr, _ = roc_curve(teste, teste2)
        roc_auc = auc(fpr, tpr)
        
        # for i in range(len(self.classes)):
        #     print(teste[i])
        # for i in range(len(self.classes)):
        #     fpr[i], tpr[i], _ = roc_curve(teste[:, i], teste2[:, i])
        #     roc_auc[i] = auc(fpr[i], tpr[i])
        
        # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        # for i, color in zip(range(len(self.classes)), colors):
        #     plt.plot(fpr[i], tpr[i], color=color, lw=2,
        #     label='ROC curve of class {0} (area = {1:0.2f})'
        #     ''.format(i, roc_auc[i]))
        # plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.legend(loc="lower right")
        # plt.show()
            

    def process(self):
        self.verifyMissValues()
        self.distributionAttributes()
        self.histogram()
        self.barCategoricalData()
        self.classBalance()
        self.pairs()

