from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd



class DataFrameSelector(BaseEstimator,TransformerMixin):

    def __init__(self):
        None
    def fit(self,X,y = None):
        return self

    def transform(self,X,y = None):
        return X.values


def Back_to_DataFrame(X,Sc,MM,Sc_X=None,MM_X=None):
        columns = X.columns.values
        if Sc and MM :
            Sc_df = pd.DataFrame(data = Sc_X,columns = columns)
            MM_df = pd.DataFrame(data = MM_X,columns = columns)
            return Sc_df,MM_df
        elif Sc and not MM :
            Sc_df = pd.DataFrame(data = Sc_X,columns = columns)
            return Sc_df
        elif MM and not Sc :
            MM_df = pd.DataFrame(data = MM_X,columns = columns)
            return MM_df




class Normalisation(BaseEstimator,TransformerMixin):

    def __init__(self,Sc = True,MM = False):
        '''Class za normalizacijo podatkov. Uporabi lahko dve metodi -->Standard Scaling ali pa Min 
           -Max normalizacijo, lahko pa tudi obedve naenkrat, odvisno od hyperparametrov.
           Vrne pa transformiran DataFrame.  
        '''
        self.Sc = Sc
        self.MM = MM
        self.Sc_ = StandardScaler()
        self.MM_ = MinMaxScaler()
        self.Std_scaler_pipeline = Pipeline([('dataFS',DataFrameSelector()),
                                        ('Std_scaler',self.Sc_),
                                    ])
        self.Min_Max_pipeline = Pipeline([ ('dataFS_cat',DataFrameSelector()),
                                       ('Min_Max_scaler', self.MM_),     
                                    ])

    def transform(self,X,y = None,transform = False):
        if self.Sc:
            Sc_X = self.Std_scaler_pipeline.fit_transform(X)
        if self.MM:
            MM_X = self.Min_Max_pipeline.fit_transform(X)
        # fit in transform podatke
        if not transform:
            if self.Sc and self.MM :
                Sc_transformed,MM_transformed = Back_to_DataFrame(X,self.Sc,self.MM,Sc_X,MM_X)
                return Sc_transformed,MM_transformed
            elif self.Sc and not self.MM:
                Sc_transformed= Back_to_DataFrame(X,self.Sc,self.MM,Sc_X)
                return Sc_transformed
            elif self.MM and not self.Sc:
                MM_transformed = Back_to_DataFrame(X,self.Sc,self.MM,None,MM_X)
                return MM_transformed
        # samo transformiraj podatke--> uporabi za test set
        else:
            if self.Sc and self.MM :
                Sc_transformed,MM_transformed = Back_to_DataFrame(X,self.Sc,self.MM,Sc_X,MM_X)
                return Sc_transformed,MM_transformed
            elif self.Sc and not self.MM :
                Sc_transformed= Back_to_DataFrame(X,self.Sc,self.MM,Sc_X)
                return Sc_transformed
            elif self.MM and not self.Sc :
                MM_transformed = Back_to_DataFrame(X,self.Sc,self.MM,None,MM_X)
                return MM_transformed

data = pd.read_csv('../data/lol.csv')
n = Normalisation()
a= n.transform(data)

print(a)