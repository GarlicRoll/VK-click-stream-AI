import pandas as pd
import time
import numpy as np
from googletrans import Translator
from scipy import spatial
import gensim.downloader as api
from gensim.models import KeyedVectors
import numpy as np
import pymystem3

import spacy

import gensim
from tqdm import tqdm
from deep_translator import GoogleTranslator
m2 = pymystem3.Mystem()
 #choose from multiple models https://github.com/RaRe-Technologies/gensim-data

model=KeyedVectors.load_word2vec_format('gensim-data/word2vec-ruscorpora-300/word2vec-ruscorpora-300.gz', binary=True)
df = pd.read_csv("train", delimiter="\t")[1000:2000]
nlp = spacy.load("ru_core_news_sm")
def make_dict(string):
    try:
        array = string.split()
        cur_dict = {array[i]:array[i+1] for i in range(0, len(array) - 1, 2)}
    except:
        cur_dict = {}
    return cur_dict
def check_eng(word):
    return word.encode().isalpha()
def translate(word):
    translator = Translator()
    word = GoogleTranslator(source='auto', target='ru').translate(word)
    return word
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
    for key, value in dict.items():
        #if check_eng(key):
            #key = translate(key) # Перевод с английского
        try:
            new_vector = get_vector(pos(key)) * np.float32(value)
        except KeyError:
            new_vector = get_vector(pos("мама")) * np.float32(value)
        if first:
            vector = new_vector
            first = False
        else:
            vector += new_vector
        
    return vector

tokens = np.array(np.zeros_like(df.tokens))

vectors = np.array(np.zeros((len(df.tokens),300)), dtype=float)

for i in tqdm(range(len(df.tokens))):
    vectors[i] = make_vector_from_dict(make_dict(df.tokens[i]))
    if i % 100 == 0:
        df_vectors = pd.DataFrame(vectors)
        df_vectors["DEF"] = df.DEF
        df_vectors.to_csv("finale3.csv")


