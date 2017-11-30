# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 20:43:53 2017

@author: patri
"""

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import time
from collections import Counter
import re
import math

def getWords():
    path = "C:\\Users\\patri\\Documents\\GW\\2017Fall\\Independent Study\\Simple project\\"
    data = pd.read_csv(path+"Data\\news_headlines.csv")
    headlines = data.drop(['URL','STORY','TIMESTAMP','HOSTNAME','ID'], axis=1)

    start = time.time()
    stopped_words = []
    i = 0
    print("Started cleaning headlines...")
    stemmer = PorterStemmer()
    for row in data['TITLE']:
        cleaned_title = re.sub('[^a-zA-Z]+',' ', row).lower()
        words = [stemmer.stem(word) for word in cleaned_title.split() if word not in stopwords.words('english')]
        stopped_words.append(','.join(words))
        i+=1
        if i % 30000 == 0:
            print("Done cleaning {} headlines".format(i))

    headlines['STOPPED_WORDS'] = stopped_words
    headlines.to_csv(path+"Data\\headline_words.csv",index=False)
    print("Took {:07.2f} seconds to create CSV file with filtered words".format(time.time()-start))
    return headlines


def getProb(pdf, words_in_cat, cat, word, total_words):
    laplace_smooth = 1
    prob = pdf.get(word)
    return ((0 if prob is None else prob) + laplace_smooth) / (words_in_cat + laplace_smooth*total_words)


class NBClassifier:
    def __init__(self, df, title_column, category_column, train_split = 1):
        train_idx = np.random.rand(len(df)) < train_split
        self.train_data = df.loc[train_idx,:]
        self.test_data = df.loc[~train_idx,:]
        self.title_column = title_column
        self.cat_column = category_column
        self.all_words = []
        self.pdf = {}
        self.words_per_cat = {}
        
        for i in range(len(df.index)):
            self.all_words += df.loc[i,self.title_column].split(',')
            if i % 50000 == 0:
                print("Done reading {} headlines".format(i))

        self.total_words = len(self.all_words)
        self.word_count = dict(Counter(self.all_words))
        self.common_words = {w:c for w,c in self.word_count.items() if c > 4}
        self.common_words.pop('')
        self.unique_words = self.common_words.keys()
        self.categories = set(self.train_data.loc[:,self.cat_column])

        self.train_accuracy = 0
        self.test_accuracy = 0

    def trainPDF(self):
        i = 1
        for cat in self.categories:
            print("Creating PDf for category {}/{}".format(i,len(self.categories)))
            indexed = self.train_data.loc[lambda df: df.loc[:,self.cat_column] == cat,:]
            self.words_per_cat[cat] = 0
            self.pdf[cat]={}
            for row in indexed.loc[:,self.title_column]:
                title_words = row.split(',')
                self.words_per_cat[cat] += len(title_words)
                for word in title_words:
                    if self.pdf[cat].get(word):
                        self.pdf[cat][word] += 1
                    else:
                        self.pdf[cat][word] = 1
            i+=1

    def predictCat(self, title, already_stopped = False):
        if already_stopped:
            words = [word for word in title.split(',')]
        else:
            stemmer = PorterStemmer()
            cleaned_title = re.sub('[^a-zA-Z]+',' ', title).lower()
            words = [stemmer.stem(word) for word in cleaned_title.split() if word not in stopwords.words('english')]
        preds = {}
        for cat in self.categories:
            preds[cat] = 0
            for word in words:
                preds[cat] += math.log(getProb(self.pdf[cat], self.words_per_cat[cat], cat, word, self.total_words))

        return preds

    def calcSummaryStats(self):
        self.train_data.loc[:,'PREDICTED'] = ''
        self.test_data.loc[:,'PREDICTED'] = ''
        for idx, row in self.train_data.iterrows():
            tr_predictions = self.predictCat(row.loc[self.title_column], True)
            self.train_data.loc[idx,'PREDICTED'] = max(tr_predictions, key = tr_predictions.get)
            if idx % 1000 == 0:
                print(idx)
            if idx > 20000:
                print("20000 done")
                break

        for idx, row in self.test_data.iterrows():
            te_predictions = self.predictCat(row.loc[self.title_column], True)
            self.test_data.loc[idx,'PREDICTED'] = max(te_predictions, key = te_predictions.get)
            if idx > 5:
                break

        self.train_accuracy = sum(self.train_data.PREDICTED == self.train_data.CATEGORY) / len(self.train_data)
        if len(self.test_data > 0):
            self.test_accuracy = sum(self.test_data.PREDICTED == self.test_data.CATEGORY) / len(self.test_data)


