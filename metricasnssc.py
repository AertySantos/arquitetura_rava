import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# === 1️  Lê ou cria o dataframe ===
output_csv = "outputs/eval_results.csv"
df = pd.DataFrame()
df = pd.read_csv(output_csv)
#df.to_csv(output_csv, index=False)

# === 2️⃣ Normaliza colunas e converte listas ===
def to_list(x):
    if pd.isna(x):
        return []
    return [i.strip() for i in str(x).split(",") if i.strip()]

df["gold_cuis"] = df["gold_cuis"].apply(to_list)
df["pred_cui"] = df["pred_cui"].apply(to_list)

# === 3️⃣ Calcula a métrica de acurácia conforme o artigo ===
def accuracy_row(row):
    gold = set(row["gold_cuis"])
    pred = set(row["pred_cui"])
    if not gold:
        return 0.0
    return len(gold.intersection(pred)) / len(gold)

df["accuracy_article"] = df.apply(accuracy_row, axis=1)

# === 4️⃣ Calcula a média geral da métrica ===
overall_accuracy = df["accuracy_article"].mean()

# === 5️⃣ Exibe resultados ===
print("\n📊 Resultados do Benchmark (métrica do artigo):")
print(f"Accuracy média: {overall_accuracy:.3f}")
print("=" * 40)

# Exibe resumo por termo
print("\n🔍 Resumo por termo:")
print(df[["term", "gold_cuis", "pred_cui", "accuracy_article"]])