import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# carregar modelo de embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")


# carregar CSV de respostas reais
df = pd.read_csv("logs/logE2E.csv", sep=";", header=0)
df["TempoSeg"] = pd.to_numeric(df["TempoSeg"], errors="coerce")

print(df)
# carregar CSV de gabarito esperado
gabarito_df = pd.read_csv("logs/gabarito.csv", sep=";",  header=0)

# unir os dois conjuntos pela pergunta
df_merged = pd.merge(df, gabarito_df, on="Entrada", how="left")
print(df_merged)
resultados = []
grupo = []
preciso = 0
falhas = 0
tipo="T3"
print(df_merged.columns)

for _, row in df_merged.iterrows():
    resp_real = str(row["Saida"])
    resp_esp = str(row["SaidaEsperada"])

    emb_r = model.encode([resp_real])
    emb_g = model.encode([resp_esp])
    sim = cosine_similarity(emb_r, emb_g)[0][0]

    if sim >= 0.5:
        preciso += 1
    elif sim < 0.4 or resp_real.strip() == "" or row['TempoSeg'] > 30:
        falhas += 1
    print(f"{resp_real}:{resp_esp}:{sim}\n")
# cálculos finais (fora do loop)

tempo_medio = df_merged["TempoSeg"].mean()
precisao = (preciso / len(df_merged)) * 100
taxa_falhas = (falhas / len(df_merged)) * 100
print(df_merged["TempoSeg"])
resultados.append({
    "Tipo de Entrada": tipo,
    "Tempo Médio (s)": round(tempo_medio, 2),
    "Precisão Semântica (%)": round(precisao, 2),
    "Taxa de Falhas (%)": round(taxa_falhas, 2)
})

print(resultados)