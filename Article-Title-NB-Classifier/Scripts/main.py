# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 17:15:44 2017

@author: patri
"""

import os
os.chdir("C:\\Users\\patri\\Documents\\GW\\2017Fall\\Independent Study\\Simple project\\Scripts")

import module3 as mod
import pandas as pd

path = "C:\\Users\\patri\\Documents\\GW\\2017Fall\\Independent Study\\Simple project\\"

titles = pd.read_csv(path+"Data\\news_headlines.csv")

if locals().get('headlines') == None:
    if "headline_words.csv" in os.listdir(path+"Data"):
        headlines = pd.read_csv(path + "Data\\headline_words.csv", encoding = "latin1", keep_default_na = False)
    else:
        headlines = mod.cleanWords(titles, path)



x = NBClassifier(headlines, train_split = 0.8)
x.trainPDF()
x.runClassifier()
x.predictCat('Apple posts higher returns this quarter')

