import joblib
import pandas as pd
import numpy as np

# Предсказание без известного ответа
# Алгоритм похож на тот, что использован, для обучения (по крайней мере предобработка даннх одинакова)

model1 = joblib.load("model1")

df = pd.read_csv("test", delimiter="\t")

df = pd.read_csv("train", delimiter="\t")

def make_dict(string):
    # Место для настройки параметров - основные ресурсы съедает преобразование стро
    try:
        array = string.split()[:20]
        cur_dict = {array[i]:array[i+1] for i in range(0, len(array) - 1, 2)}
    except:
        cur_dict = {}
    
    sorted_big_dict = {}
    sorted_keys = sorted(cur_dict, key=cur_dict.get, reverse=True)

    for w in sorted_keys:
        sorted_big_dict[w] = cur_dict[w]

    i = 0
    final_dict = {}
    for key, value in sorted_big_dict.items():
        if i == 10:
            break
        i += 1
        final_dict[key] = value 
    
    return final_dict

def make_final_dict(field):
    iters = 0
    iters_i = 0
    big_dict = {}
    for i in range(len(field)):
        token_dict = make_dict(field[i])
        iters_i += 1
        for key, arg in token_dict.items():
            iters += 1
            if key in big_dict:
                big_dict[key] += int(arg)
                pass            
            else:
                big_dict[key] = 1
    sorted_big_dict = {}
    sorted_keys = sorted(big_dict, key=big_dict.get, reverse=True)

    for w in sorted_keys:
        sorted_big_dict[w] = big_dict[w]

    i = 0
    final_dict = {}
    for key, value in sorted_big_dict.items():
        if i == 1000:
            break
        i += 1
        final_dict[key] = value 
    return final_dict

def fill_dataframe(df, field, final_dict):
    for i in range(len(field)): 
        for key, arg in final_dict.items():
            try:
                user_dict = make_dict(field[i])
                if key in user_dict:
                    df.loc[i, key] += np.float64(user_dict[key])
            except KeyError:
                pass
    return df

tokens_dict = make_final_dict(df.tokens)
hashes_dict = make_final_dict(df.urls_hashed)

df_tokens = pd.DataFrame(np.zeros((len(df.tokens), len(list(tokens_dict.keys())))), columns=list(tokens_dict.keys()))
df_hashes = pd.DataFrame(np.zeros((len(df.tokens), len(list(hashes_dict.keys())))), columns=list(hashes_dict.keys()))
df = pd.concat([df, df_tokens, df_hashes], axis="columns")

df = fill_dataframe(df, df.tokens, tokens_dict)
df = fill_dataframe(df, df.urls_hashed, hashes_dict)

X = df.drop(["urls_hashed", "tokens", "DEF", "CLIENT_ID", "RETRO_DT"], axis="columns")

Y = model1.predict(X)

# Создание файла csv с предсказаниями
df_final = pd.DataFrame(Y)
df_final = pd.concat([df_final, df.CLIENT_ID, df.RETRO_DT], axis="columns")

df_final.to_csv("finale_model1.csv")