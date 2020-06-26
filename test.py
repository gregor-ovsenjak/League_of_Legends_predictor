import re
import numpy as np
import pandas as pd

data = pd.read_csv('data/lol.csv')
data = data.drop('blueWins',axis =1)
regex_expresion = r'^blue(.*?[^Diff])$'
print(data.columns.values)
data2 = pd.DataFrame()

for col_name in data.columns.values:

    regex_result = re.findall(regex_expresion,col_name)
    if regex_result:
        red_col_name = 'red'+ regex_result[0]
        data2['BRdiff'+regex_result[0]] = data['blue'+regex_result[0]] - data[red_col_name]
print(data2.columns.values)