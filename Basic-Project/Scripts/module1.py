# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 20:43:53 2017

@author: patri
"""

import pandas as pd
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


def getAllWords(df):
    combined_list = []
    for i in range(len(df.index)):
        combined_list += df['STOPPED_WORDS'][i].split(',')
        if i % 50000 == 0:
            print("Done processing {} headlines".format(i))
    return combined_list


def createPDF(df,words):
    pdf = {}
    num_words = {}
    categories = ["b","t","e","m"]
    i = 1
    for cat in categories:
        print("Creating PDf for category {}/{}".format(i,len(categories)))
        indexed = df.loc[lambda df: df.CATEGORY == cat,:]
        num_words[cat] = 0
        pdf[cat]={}
        for row in indexed['STOPPED_WORDS']:
            title_words = row.split(',')
            num_words[cat] += len(title_words)
            for word in title_words:
                if pdf[cat].get(word):
                    pdf[cat][word] += 1
                else:
                    pdf[cat][word] = 1
        i+=1
    return pdf, num_words


def getProb(pdf, num_words, cat, word, total_words):
    laplace_smooth = 1
    if pdf[cat].get(word):
        return (pdf[cat][word] + laplace_smooth) / (num_words[cat] + laplace_smooth*total_words)
    else:
        return laplace_smooth / (num_words[cat] + laplace_smooth*total_words)


def predictCat(title):
    categories = ["b", "t", "e", "m"]    
    stemmer = PorterStemmer()
    cleaned_title = re.sub('[^a-zA-Z]+',' ', title).lower()
    words = [stemmer.stem(word) for word in cleaned_title.split() if word not in stopwords.words('english')]
    preds = {}
    for cat in categories:
        preds[cat] = 0
        for word in words:
            preds[cat] += math.log(getProb(pdf, num_words, cat, word, total_words))
    
    return max(preds.keys(), key = (lambda key: preds[key]))



class NBClassifier:
    def __init__(self, df, title_column, category_column):
        self.data = df
        self.title_column = title_column
        self.cat_column = category_column
        self.all_words = []
        self.pdf = {}
        self.words_per_cat = {}
        
        for i in range(len(df.index)):
            self.all_words += df[self.title_column][i].split(',')
            if i % 50000 == 0:
                print("Done reading {} headlines".format(i))

        self.word_count = dict(Counter(self.all_words))
        self.common_words = {w:c for w,c in self.word_count.items() if c > 4}
        self.common_words.pop('')
        self.unique_words = self.common_words.keys()
        self.categories = set(self.data[self.cat_column])

    def createPDF(self):
        i = 1
        for cat in self.categories:
            print("Creating PDf for category {}/{}".format(i,len(self.categories)))
            indexed = self.data.loc[lambda df: df[self.cat_column] == cat,:]
            self.words_per_cat[cat] = 0
            self.pdf[cat]={}
            for row in indexed[self.title_column]:
                title_words = row.split(',')
                self.words_per_cat[cat] += len(title_words)
                for word in title_words:
                    if self.pdf[cat].get(word):
                        self.pdf[cat][word] += 1
                    else:
                        self.pdf[cat][word] = 1
            i+=1

    def predictCat(self,title):
        stemmer = PorterStemmer()
        cleaned_title = re.sub('[^a-zA-Z]+',' ', title).lower()
        words = [stemmer.stem(word) for word in cleaned_title.split() if word not in stopwords.words('english')]
        preds = {}
        for cat in self.categories:
            preds[cat] = 0
            for word in words:
                preds[cat] += math.log(getProb(self.pdf, self.words_per_cat, cat, word, self.all_words))
        
        return preds
        
x = NBClassifier(headlines, 'STOPPED_WORDS','CATEGORY')