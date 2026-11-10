import os
import sys
from datetime import datetime

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
diretorio_raiz = os.path.abspath(os.path.join(diretorio_atual, ".."))
sys.path.append(diretorio_raiz)

import tensorflow as tf
from data.planilha.cleaning.limpeza_planilha_classificacao import carregar_e_preparar_dados


def obter_callback_tensorboard(diretorio_base="outputs"):
    momento_exato = datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_log = os.path.join(diretorio_base, "logs_tensorboard_classificacao", momento_exato)

    os.makedirs(dir_log, exist_ok=True)

    return tf.keras.callbacks.TensorBoard(
        log_dir=dir_log,
        histogram_freq=1
    )

def treinar_modelo_classificacao():
    x_train, y_train, x_test, y_test, norm_camada, colunas = carregar_e_preparar_dados()

    #Rede Neural simples, baseada no modelo Sequencial, mas com Dropout para regularização
    modelo_tipo = tf.keras.Sequential()
    modelo = modelo_tipo

    modelo.add(norm_camada)
    modelo.add(tf.keras.layers.Dense(128, activation="relu", name="camada_oculta_1"))
    modelo.add(tf.keras.layers.Dropout(0.25))  # Levemente mais forte para a camada maior
    modelo.add(tf.keras.layers.Dense(64, activation="relu", name="camada_oculta_2"))
    modelo.add(tf.keras.layers.Dropout(0.2))
    modelo.add(tf.keras.layers.Dense(32, activation="relu", name="camada_oculta_3"))
    modelo.add(tf.keras.layers.Dropout(0.2))
    modelo.add(tf.keras.layers.Dense(1, activation="sigmoid", name="camada_saida_classificacao"))

    #Compilador do modelo
    compilacao_modelo = modelo.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001,

        ),

        loss=tf.keras.losses.BinaryCrossentropy(
            reduction="sum_over_batch_size", name="binary_crossentropy", dtype=None
        ),

        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy")]
    )

    callbacks_lista = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15, 
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,    # Reduz o learning_rate pela metade quando estagnar
            patience=5,    # Espera 5 épocas sem melhora para agir
            min_lr=1e-6    # Limite mínimo para o learning_rate
        ),
        obter_callback_tensorboard()
    ]

    #Treinamento do modelo
    modelo_treinamento = modelo.fit(
        x_train,
        y_train,
        epochs=150,
        batch_size=32,
        validation_data=(x_test,y_test),
        callbacks=callbacks_lista
    )

    #Avaliação final, comparando a predição e os valores de teste
    avaliacao_modelo = modelo.evaluate(
        x_test,
        y_test,
        batch_size=32
    )

    modelo.save("outputs/modelo_viral_classificacao.keras")

    return modelo_treinamento, avaliacao_modelo

if __name__ == "__main__":
    print("Iniciando o pipeline de treinamento...")
    
    historico_treino, avaliacao_final = treinar_modelo_classificacao()
    
    # Imprime os resultados da avaliação (Loss e Binary) do conjunto de teste
    print("\n--- Treinamento Concluído ---")
    print(f"Perda (Loss) no Teste: {avaliacao_final[0]:.6f}")
    print(f"Taxa de Acerto no Teste: {avaliacao_final[1] * 100:.2f}%")