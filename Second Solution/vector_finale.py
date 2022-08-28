import pandas as pd
import time
import numpy as np
from googletrans import Translator
from scipy import spatial
import gensim.downloader as api
from gensim.models import KeyedVectors
import numpy as np
import pymystem3
from scipy import spatial
import spacy

import gensim

from deep_translator import GoogleTranslator


from tqdm import tqdm

m2 = pymystem3.Mystem()
 #choose from multiple models https://github.com/RaRe-Technologies/gensim-data

model=KeyedVectors.load_word2vec_format('gensim-data/word2vec-ruscorpora-300/word2vec-ruscorpora-300.gz', binary=True)

nlp = spacy.load("ru_core_news_sm")

def make_dict(string):
    try:
        array = string.split()
        cur_dict = {array[i]:array[i+1] for i in range(0, len(array) - 1, 2)}
    except:
        cur_dict = {}
    return cur_dict

def pos(s0):
    s0=m2.lemmatize(s0) # Переход к единственному числу и именительному падежу
    s0=list(filter(lambda x: x != ' ', s0))
    s0=list(filter(lambda x: x != '\n', s0))
    for s in range(len(s0)):
        document=nlp(s0[s])
        s0[s]=s0[s]+'_'+document[0].pos_ # Добавляется часть речи
    return ' '.join(s0)

def preprocess(s):
    return [i for i in s.split()]

def get_vector(s):
    return np.sum(np.array([model[i] for i in preprocess(s)]), axis=0)

def make_vector_from_dict(dict):
    first = True
    default_vector = get_vector(pos("мама")) 
    vector = default_vector
    for key, value in dict.items():
        #if check_eng(key):
            #key = translate(key) # Перевод с английского
        try:
            new_vector = get_vector(pos(key)) * np.float32(value)
        except KeyError:
            new_vector = default_vector * np.float32(value)
        if first:
            vector = new_vector
            first = False
        else:
            vector += new_vector
        
    return vector

df_train = pd.read_csv("finale2.csv")[:50]

df_test = pd.read_csv("test", sep="\t")

X = df_train.drop(['DEF'], axis="columns")

X = X.drop(["Unnamed: 0"], axis="columns")

results = np.array(np.zeros((len(df_test.tokens))))
for j in tqdm(range(len(df_test.tokens))):
    vector = make_vector_from_dict(make_dict(df_test.tokens[j]))
    df_cosine = pd.DataFrame([1 - spatial.distance.cosine(X.iloc[i], vector) for i in range(len(X))])
    index = df_cosine.sort_values(0 ,ascending=False).iloc[0].name
    results[j] = df_train.DEF[index]
    if j % 10 == 0:
        df_results = pd.DataFrame({"DEF": results})
        df_results = pd.concat([df_results, df_test["CLIENT_ID"], df_test["RETRO_DT"]], axis="columns")
        df_results.to_csv("finale_vectors.csv")