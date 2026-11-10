import pandas as pd
import os

diretorio_atual = os.path.dirname(os.path.abspath(__file__))
caminho_csv = os.path.join(diretorio_atual, "..","csv")

df_audio = pd.read_csv(os.path.join(caminho_csv, "audio_features.csv"))
df_meta = pd.read_csv(os.path.join(caminho_csv, "track_metadata.csv"))
df_chart = pd.read_csv(os.path.join(caminho_csv, "chart_longevity_panel.csv" ))

df_consolidado = df_audio.merge(df_meta, on="track_id").merge(df_chart, on="track_id")

caminho_saida = os.path.join(diretorio_atual, "spotify_datas_consolidado.xlsx")

df_consolidado.to_excel(caminho_saida, index=False)
print("Planilha consolidada criada com sucesso! Total de colunas:", len(df_consolidado.columns))



