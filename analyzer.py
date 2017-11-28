import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter

class Analyzer(object):

    def readFiles(self):
        self.train_set = pd.read_csv('train.csv', header=0, index_col=None)
        return

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
        labels = self.train_set.drop(['id','bone_length','hair_length','has_soul','color'], axis=1)
        colors = ("red", "green", "blue")
        area = np.pi*2
        x = labels['type']
        y = labels['rotting_flesh']
        plt.scatter(x, y, s=area, c=colors)
        plt.show()
    

analyzer = Analyzer()
analyzer.readFiles()
analyzer.histogram()
analyzer.barCategoricalData()
analyzer.classBalance()
#analyzer.distributionAttributes()
