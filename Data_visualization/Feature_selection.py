import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from itertools import cycle, islice



# function for seperating target values and feature values
def seperate_target_feature(data,target_name):
    y = data[target_name]
    X = data.drop(target_name,axis = 1)
    return X,y



class FeatImp():

    def __init__(self,data,classifier = ExtraTreesClassifier()):
        self.data = data
        self.model = classifier # tree based classifiers are used for visualizing Feature importance



    def visualize(self,target_name):
        # plots a horizontal bar graph of most important features, selected by Extra
        # TreesClassifier. You can choose your own classifier as long as it is tree based
        X,y = seperate_target_feature(self.data,target_name)
        self.model.fit(X,y)
        # colors of the bars 
        my_colors = list(islice(cycle(['blue', 'red', 'green', 'gold', 'm','olive','purple','cyan','maroon']), None, len(X)))
        
        # plot graph of feature importances for better visualization
        feat_importances = pd.Series(self.model.feature_importances_, index=X.columns)
        feat_importances.nlargest(10).plot(kind='barh',stacked=True,color = my_colors)
        plt.show()
    
    def corr_matrix(self,target_name):
        X,y = seperate_target_feature(self.data,target_name)
        #get correlations of each features in dataset
        correlation_matrix = self.data.corr()
        top_corr_features = correlation_matrix[target_name].nlargest(10).index.values
        print(top_corr_features)
        plt.figure(figsize=(10,10))
        #plot heat map
        g=sns.heatmap(self.data[top_corr_features].corr(),annot=True,cmap="RdYlGn")
        plt.show()


featimp = FeatImp(data = pd.read_csv('../data/lol.csv'))
featimp.corr_matrix('blueWins')
    
