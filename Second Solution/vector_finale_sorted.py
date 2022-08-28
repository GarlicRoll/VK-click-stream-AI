#!!!! Обязательно к установке после requirements.txt
# python -m gensim.downloader --download word2vec-ruscorpora-300
# python -m spacy download ru_core_news_sm

import pandas as pd
import time
import numpy as np
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

# Лемматизация - преобразование к ед.числу и им.падежу
m2 = pymystem3.Mystem()

# Заранее подготовленная NLP библиотека 
# Пример визуализации https://rusvectores.org/ru/ruscorpora_upos_cbow_300_20_2019/%D1%81%D0%BE%D0%BB%D0%BD%D1%86%D0%B5_PROPN/
model=KeyedVectors.load_word2vec_format('gensim-data/word2vec-ruscorpora-300/word2vec-ruscorpora-300.gz', binary=True)

nlp = spacy.load("ru_core_news_sm")

# Преобразование строки в словарь
# Формата {key:value,...,...}
def make_dict(string):
    try:
        array = string.split()
        cur_dict = {array[i]:array[i+1] for i in range(0, len(array) - 1, 2)}
    except:
        cur_dict = {}
    
    sorted_big_dict = {}
    sorted_keys = sorted(cur_dict, key=cur_dict.get, reverse=True)

    for w in sorted_keys:
        sorted_big_dict[w] = cur_dict[w]

    i = 0
    sorted_big_dict=sorted_big_dict
    final_dict = {}
    for key, value in sorted_big_dict.items():
        if i == 5:
            break
        i += 1
        final_dict[key] = value 
        
    return final_dict

# ruscorpora требует, чтобы формат запроса был: "словосмаленькойбуквы_ЧАСТЬРЕЧИНААНГЛИЙСКОМ"
def pos(s0):
    s0=m2.lemmatize(s0) # Переход к единственному числу и именительному падежу
    s0=list(filter(lambda x: x != ' ', s0))
    s0=list(filter(lambda x: x != '\n', s0))
    for s in range(len(s0)):
        document=nlp(s0[s])
        s0[s]=s0[s]+'_'+document[0].pos_ # Добавляется часть речи
    return ' '.join(s0)

# Сплит строки на array
def preprocess(s):
    return [i for i in s.split()]

# Превращение массива в одномерный 1-D вектор
def get_vector(s):
    return np.sum(np.array([model[i] for i in preprocess(s)]), axis=0)

# Преобразование словаря в вектора
def make_vector_from_dict(dict):
    first = True
    default_vector = get_vector(pos("мама")) 
    vector = default_vector
    for key, value in dict.items():
        #if check_eng(key):
            #key = translate(key) # Перевод с английского
        try:
            #Получаем значение вектора и домножаем на количество упоминаний (чем больше упоминаний = тем больше значимость вектора)
            #Получаем итоговый вектор
            new_vector = get_vector(pos(key)) * np.float32(value)
        except KeyError:
            new_vector = default_vector * np.float32(value)
        if first:
            vector = new_vector
            first = False
        else:
            vector += new_vector
        
    return vector

# Загружаем готовый файл с векторами на основе train файла
df_train = pd.read_csv("finale2.csv")[:20]

df_test = pd.read_csv("test", sep="\t")

X = df_train.drop(['DEF'], axis="columns")

X = X.drop(["Unnamed: 0"], axis="columns")

# Предобработка test файла
results = np.array(np.zeros((len(df_test.tokens))))

# Цикл, проходящий все tokens в test файле
for j in tqdm(range(len(df_test.tokens))):
    # Преобразование tokens в одномерные 1-D вектора
    vector = make_vector_from_dict(make_dict(df_test.tokens[j]))
    # Вычисление ближайшего вектора к j-му значению tokens в test
    df_cosine = pd.DataFrame([1 - spatial.distance.cosine(X.iloc[i], vector) for i in range(len(X))])
    index = df_cosine.sort_values(0 ,ascending=False).iloc[0].name
    # Присваивание целевого показателя строке test на основе ближайшего вектора
    results[j] = df_train.DEF[index]
    # Сохранение обработанных строк каждые 10 иттераций
    if j % 10 == 0:
        # Форматирование и экспорт в требуемом формате
        df_results = pd.DataFrame({"DEF": results})
        df_results = pd.concat([df_results, df_test["CLIENT_ID"], df_test["RETRO_DT"]], axis="columns")
        df_results.to_csv("finale_vectors_sorted.csv")