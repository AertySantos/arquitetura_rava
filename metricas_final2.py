import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np

# carregar modelo de embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# carregar CSV de respostas reais
df = pd.read_csv("logs/logE2E.csv", sep=";", header=0)
df["TempoSeg"] = pd.to_numeric(df["TempoSeg"], errors="coerce").fillna(0)

# carregar CSV de gabarito esperado
gabarito_df = pd.read_csv("logs/gabarito.csv", sep=";", header=0)


def embedding_merge(df1, df2, key1, key2, threshold=0.90):
    """
    Faz merge aproximado entre df1[key1] e df2[key2] usando embeddings + cosine similarity
    """
    emb1 = model.encode(df1[key1].astype(str).tolist())
    emb2 = model.encode(df2[key2].astype(str).tolist())

    sims = cosine_similarity(emb1, emb2)

    matches = []
    for i, row in enumerate(sims):
        j = np.argmax(row)  # índice do gabarito mais parecido
        score = row[j]
        if score >= threshold:
            matches.append((df2.iloc[j][key2], score))
        else:
            matches.append((None, score))

    df1["match"] = [m[0] for m in matches]
    df1["Similaridade_Entrada"] = [m[1] for m in matches]

    result = pd.merge(df1, df2, left_on="match", right_on=key2, how="left")
    result = result.dropna(subset=["match"])  # remove linhas sem match
    return result


def calcular_metricas(df_tipo, gabarito_df, tipo):
    if df_tipo.empty:
        return None
    
    df_merged = embedding_merge(
        df_tipo.copy(), gabarito_df, "Entrada", "Entrada", threshold=0.80
    )

    preciso = 0
    parcial = 0
    falhas = 0

    # embeddings das respostas
    emb_real = model.encode(df_merged["Saida"].astype(str).tolist())
    emb_esp = model.encode(df_merged["SaidaEsperada"].astype(str).tolist())

    sims = cosine_similarity(emb_real, emb_esp).diagonal()
    df_merged["Similaridade_Resposta"] = sims

    classificacoes = []

    for i, row in enumerate(df_merged.itertuples()):
        resp_real = str(row.Saida).strip()
        sim = sims[i]
        tempo = getattr(row, "TempoSeg", 0) or 0

        if sim < 0.4 or resp_real == "" or tempo > 300:
            falhas += 1
            classificacoes.append("Falha")
        elif sim >= 0.65:
            preciso += 1
            classificacoes.append("Preciso")
        else:
            parcial += 1
            classificacoes.append("Parcial")

    df_merged["Classificacao"] = classificacoes

    total = len(df_merged)
    tempo_medio = df_merged["TempoSeg"].mean() if total > 0 else 0
    taxa_preciso = (preciso / total * 100) if total > 0 else 0
    taxa_parcial = (parcial / total * 100) if total > 0 else 0
    taxa_falhas = (falhas / total * 100) if total > 0 else 0

    # salvar resultados detalhados
    df_merged.to_csv(f"metricas_{tipo}.csv", sep=";", index=False, encoding="utf-8")

    return {
        "tipo": tipo,
        "tempo": round(tempo_medio, 2),
        "precisao": round(taxa_preciso, 2),
        "parcial": round(taxa_parcial, 2),
        "falhas": round(taxa_falhas, 2),
    }


# calcular métricas por tipo
resultados = []
for tipo in ["T1", "T2", "T3"]:
    df_tipo = df[df["TipoEntrada"] == tipo]
    metrica = calcular_metricas(df_tipo, gabarito_df, tipo)
    if metrica:
        resultados.append(metrica)

# salvar resumo
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv("metricas_resumo.csv", sep=";", index=False, encoding="utf-8")

print("✅ Arquivos gerados: metricas_T1.csv, metricas_T2.csv, metricas_T3.csv e metricas_resumo.csv")
