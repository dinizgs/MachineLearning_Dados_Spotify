import os
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

def carregar_e_preparar_dados(caminho_planilha=None):
    if caminho_planilha is None:
        diretorio_atual = os.path.dirname(os.path.abspath(__file__))
        caminho_planilha = os.path.join(diretorio_atual,"..","spotify_datas_consolidado.xlsx")

    df = pd.read_excel(caminho_planilha)

    df_sem_duplicata = df.drop_duplicates(subset="track_id")
    df_limpo = df_sem_duplicata.dropna(axis=1, how="all")

    #Transformar as variáveis categóricas em numéricas com One-Hot-Encoder
    colunas_categorias = ["genre","musical_key","mode"]
    df_encoded = pd.get_dummies(df_limpo,columns=colunas_categorias, dtype=float)

    #Colunas para descartar
    colunas_para_descartar = [
        "track_id",
        "track_title",
        "artist_name",
        "release_date",
        "is_viral_hit"
    ]

    #Classificação
    X = df_encoded.drop(columns=colunas_para_descartar)
    y = df_encoded["is_viral_hit"]

    x_array = X.values.astype("float32")
    y_array = y.values.astype("float32")

    tamanho_total = len(x_array)
    indices = np.arange(tamanho_total)
    np.random.seed(42)
    np.random.shuffle(indices)

    x_array = x_array[indices]
    y_array = y_array[indices]

    # 80% dados dos para treino e 20% para testes
    limite_treino = int(tamanho_total * 0.8)

    x_train = x_array[:limite_treino]
    y_train = y_array[:limite_treino]

    x_test = x_array[limite_treino:]
    y_test = y_array[limite_treino:]

    norm_camada = layers.Normalization()
    norm_camada.adapt(x_train)

    return x_train, y_train, x_test, y_test, norm_camada, X.columns