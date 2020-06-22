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
        my_colors = list(islice(cycle(['b', 'r', 'g', 'gold', 'm','olive','purple','cyan','maroon']), None, len(X)))
        
        # plot graph of feature importances for better visualization
        feat_importances = pd.Series(self.model.feature_importances_, index=X.columns)
        feat_importances.nlargest(10).plot(kind='barh',stacked=True,color = my_colors)
        plt.show()



    
    
