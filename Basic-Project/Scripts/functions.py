# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 20:43:53 2017

@author: patri
"""

import pandas as pd
from nltk.corpus import stopwords
import time
from collections import Counter
import re

def getWords():
    path = "C:\\Users\\patri\\Documents\\GW\\2017Fall\\Independent Study\\Simple project\\"
    data = pd.read_csv(path+"Data\\news_headlines.csv")
    headlines = data.drop(['URL','STORY','TIMESTAMP','HOSTNAME','ID'], axis=1)
    
    start = time.time()
    stopped_words = []
    i = 0
    print("Started cleaning headlines...")
    for row in data['TITLE']:
        cleaned_title = re.sub('[^a-zA-Z]+',' ', row).lower()
        words = [word for word in cleaned_title.split() if word not in stopwords.words('english')]
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
    categories = ["b","t","e","m"]
    i = 1
    for cat in categories:
        print("Creating PDf for category {}/{}".format(i,len(categories)))
        indexed = df.loc[lambda df: df.CATEGORY == cat,:]
        all_words = []
        for row in indexed['STOPPED_WORDS']:
            all_words + row.split(',')
        print(all_words[:30])
        pdf["num words in category "+cat] = len(all_words)
        pdf[cat]={}
        for word in words:
            pdf[cat][word] = all_words.count(word)
        i+=1
    return pdf


        
        
