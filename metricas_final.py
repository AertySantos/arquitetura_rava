import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# carregar modelo de embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# carregar CSV de respostas reais
df = pd.read_csv("logs/logE2E.csv", sep=";", header=0)
df["TempoSeg"] = pd.to_numeric(df["TempoSeg"], errors="coerce")

# carregar CSV de gabarito esperado
gabarito_df = pd.read_csv("logs/gabarito.csv", sep=";", header=0)

def calcular_metricas(df_tipo, gabarito_df, tipo):
    if df_tipo.empty:
        return None
    
    # unir com gabarito
    df_merged = pd.merge(df_tipo, gabarito_df, on="Entrada", how="left")

    preciso = 0
    falhas = 0

    # embeddings em lote
    emb_real = model.encode(df_merged["Saida"].astype(str).tolist())
    emb_esp = model.encode(df_merged["SaidaEsperada"].astype(str).tolist())

    sims = cosine_similarity(emb_real, emb_esp).diagonal()
    df_merged["Similaridade"] = sims

    for i, row in enumerate(df_merged.itertuples()):
        resp_real = str(row.Saida)
        sim = sims[i]

        if sim >= 0.65:
            preciso += 1
        elif sim < 0.4 or resp_real.strip() == "" or row.TempoSeg > 30:
            falhas += 1

    # métricas
    total = len(df_merged)
    tempo_medio = df_merged["TempoSeg"].mean()
    precisao = (preciso / total * 100) if total > 0 else 0
    taxa_falhas = (falhas / total * 100) if total > 0 else 0

    # salvar resultados detalhados por entrada
    df_merged.to_csv(f"metricas_{tipo}.csv", sep=";", index=False, encoding="utf-8")

    return {
        "Tipo de Entrada": tipo,
        "Tempo Médio (s)": round(tempo_medio, 2),
        "Precisão Semântica (%)": round(precisao, 2),
        "Taxa de Falhas (%)": round(taxa_falhas, 2)
    }

# calcular métricas por tipo
resultados = []
for tipo in ["T1", "T2", "T3"]:
    df_tipo = df[df["TipoEntrada"] == tipo]
    metrica = calcular_metricas(df_tipo, gabarito_df, tipo)
    if metrica:
        resultados.append(metrica)

# salvar resumo geral
df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv("metricas_resumo.csv", sep=";", index=False, encoding="utf-8")

print("✅ Arquivos gerados: metricas_T1.csv, metricas_T2.csv, metricas_T3.csv e metricas_resumo.csv")
