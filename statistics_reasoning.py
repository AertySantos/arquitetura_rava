import os
import re
import glob
import unidecode
import duckdb
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f_oneway

# BERTopic
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import umap
from nltk.corpus import stopwords

# Importa sua classe existente
from consultaCsv import ConsultaCSV

# baixar stopwords (só na primeira vez)
nltk.download("stopwords")


class Statistic_reasoning:
    def __init__(self, pasta_csv="csv", pasta_txt="testes/dataset/furtoSimples"):
        # inicializa ConsultaCSV
        self.consulta = ConsultaCSV(pasta_csv)
        self.pasta_txt = pasta_txt

        # stopwords
        self.stop_words = set(stopwords.words("portuguese"))
        palavras_extras = [
            "réu", "penal", "pena", "fls", "código", "paulo",
            "crime", "vítima", "acusado", "furto", "artigo", "santos"
        ]
        self.stop_words.update(palavras_extras)

    # -------------------------------
    # Remoção de stopwords
    # -------------------------------
    def remover_stopwords(self, texto):
        palavras = texto.split()
        palavras_filtradas = [p for p in palavras if p.lower() not in self.stop_words]
        return " ".join(palavras_filtradas)

    # -------------------------------
    # Função de Tópicos (BERTopic)
    # -------------------------------
    def topicos(self):
        documentos = []
        for arquivo in os.listdir(self.pasta_txt):
            if arquivo.endswith(".txt"):
                with open(os.path.join(self.pasta_txt, arquivo), "r", encoding="utf-8") as f:
                    texto = f.read()
                    documentos.append(self.remover_stopwords(texto))

        print(f"Total de documentos carregados: {len(documentos)}")

        if len(documentos) < 5:
            print("⚠️ Atenção: poucos documentos, ajuste os parâmetros de UMAP/HDBSCAN.")

        embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda:1")

        umap_model = umap.UMAP(
            n_neighbors=min(5, len(documentos)-1),
            n_components=min(5, len(documentos)-1)
        )

        vectorizer_model = CountVectorizer(stop_words=list(self.stop_words))

        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            vectorizer_model=vectorizer_model,
            language="portuguese",
            calculate_probabilities=True
        )

        topics, probs = topic_model.fit_transform(documentos)

        print("\n--- Tópicos encontrados ---")
        print(topic_model.get_topic_info())

        if len(set(topics)) > 0:
            print("\n--- Palavras do tópico 0 ---")
            print(topic_model.get_topic(0))

        topic_model.save("modelo_bertopic", save_embedding_model=True)

        topic_info = topic_model.get_topic_info()

        with open("topicos_resumidos.txt", "w", encoding="utf-8") as f:
            for index, row in topic_info.iterrows():
                f.write(f"Tópico {row['Topic']} ({row['Count']} documentos): {row['Name']}\n")

        with open("palavras_por_topico.txt", "w", encoding="utf-8") as f:
            topic_ids = topic_model.get_topic_info()["Topic"].tolist()
            for topic_id in topic_ids:
                if topic_id == -1:
                    continue
                palavras = topic_model.get_topic(topic_id)
                palavras_str = ", ".join([p[0] for p in palavras])
                f.write(f"Tópico {topic_id}: {palavras_str}\n")

        print("✅ Todas as palavras dos tópicos foram salvas em 'palavras_por_topico.txt'.")

        return topic_info

    # -------------------------------
    # Estatísticas Básicas
    # -------------------------------
    def estatisticas_basicas(self, id: int, pergunta: str) -> dict:
        #print(pergunta)
        df_resultado = self.consulta.respostaLlm(id, pergunta)
        print(df_resultado)
        print("teste")
        df_resultado = df_resultado.apply(pd.to_numeric, errors="coerce")
        df_resultado = df_resultado.fillna(0.0)
        print(df_resultado)
        
        if df_resultado.empty:
            return {"erro": "Nenhum dado encontrado para a pergunta."}

        try:
            colunas_numericas = df_resultado.select_dtypes(include=[np.number]).columns
            if len(colunas_numericas) == 0:
                return {"erro": "Nenhuma coluna numérica encontrada no resultado."}

            coluna = colunas_numericas[0]
            valores = df_resultado[coluna].dropna().astype(float)

            if valores.empty:
                return {"erro": "Nenhum valor numérico válido para calcular estatísticas."}

            estatisticas = {
                "coluna": coluna,
                "media": float(valores.mean()),
                "mediana": float(valores.median()),
                "desvio_padrao": float(valores.std()),
                "minimo": float(valores.min()),
                "maximo": float(valores.max()),
                "quantidade": int(valores.count())
            }
            return estatisticas

        except Exception as e:
            return {"erro": f"Falha ao calcular estatísticas: {str(e)}"}

    def resumo_estatisticas(self, id: int, pergunta: str) -> str:
        stats = self.estatisticas_basicas(id, pergunta)
        if "erro" in stats:
            return f"⚠️ {stats['erro']}"

        resumo = (
            f"📊 Estatísticas para a coluna **{stats['coluna']}**:\n"
            f"- Média: {stats['media']:.2f}\n"
            f"- Mediana: {stats['mediana']:.2f}\n"
            f"- Desvio padrão: {stats['desvio_padrao']:.2f}\n"
            f"- Valor mínimo: {stats['minimo']:.2f}\n"
            f"- Valor máximo: {stats['maximo']:.2f}\n"
            f"- Quantidade de valores: {stats['quantidade']}"
        )

        # Integra com o verificaResp do ConsultaCSV
        return self.consulta.verificaResp(pergunta, resumo)


    def to_float_str_or_zero(self, x):
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return 0.0
        try:
            return float(x)  # número -> float
        except (ValueError, TypeError):
            return str(x)    # texto válido -> string

    # -------------------------------
    # ANOVA + Gráfico
    # -------------------------------
    

    def to_float_or_zero_if_pena(self, x, coluna=None):
        """Converte para float apenas se a coluna for Pena_Base ou Pena_Definitiva"""
        if coluna in ["Pena_Base", "Pena_Definitiva"]:
            if x is None or (isinstance(x, str) and x.strip() == ""):
                return 0.0
            try:
                return float(x)
            except (ValueError, TypeError):
                return 0.0
        else:
            # comportamento anterior
            if x is None or (isinstance(x, str) and x.strip() == ""):
                return 0.0
            try:
                return float(x)
            except (ValueError, TypeError):
                return str(x)

    def normalizar_dataframe(self, resultado):
        """Garante que o resultado seja DataFrame e aplica conversão correta"""
        if isinstance(resultado, pd.DataFrame):
            df = resultado.copy()
            for col in df.columns:
                df[col] = df[col].apply(lambda x: self.to_float_or_zero_if_pena(x, coluna=col))
            return df
        elif isinstance(resultado, (list, dict)):
            df = pd.DataFrame(resultado)
            for col in df.columns:
                df[col] = df[col].apply(lambda x: self.to_float_or_zero_if_pena(x, coluna=col))
            return df
        else:  # string, int, float, None...
            return pd.DataFrame([resultado], columns=["resposta"]).applymap(self.to_float_or_zero_if_pena)

    def grafico_anova(self, id: int, pergunta: str, coluna_grupo: str):
        """
        Gera gráfico ANOVA entre grupos de uma coluna categórica.
        Ex: comparar Pena_Definitiva por Categoria
        """
        # garante DataFrame e conversão correta
        df_resultado = self.normalizar_dataframe(self.consulta.respostaLlm(id, pergunta))
        print(df_resultado)
        df_resultado.to_csv("resultadopp.csv", index=False)
        # verifica se coluna de grupo existe
        if coluna_grupo not in df_resultado.columns:
            print(f"⚠️ Coluna de grupo '{coluna_grupo}' não encontrada.")
            return

        # verifica colunas numéricas
        colunas_numericas = df_resultado.select_dtypes(include=[np.number]).columns
        if len(colunas_numericas) == 0:
            print("⚠️ Nenhuma coluna numérica encontrada para ANOVA.")
            return

        coluna_valor = colunas_numericas[0]

        # separa grupos
        grupos = [
            grupo[coluna_valor].dropna().astype(float).values
            for _, grupo in df_resultado.groupby(coluna_grupo)
        ]

        if len(grupos) < 2:
            print("⚠️ ANOVA requer pelo menos 2 grupos.")
            return

        # Executa ANOVA
        f_stat, p_valor = f_oneway(*grupos)
        print(f"📊 Resultado ANOVA: F = {f_stat:.4f}, p = {p_valor:.4f}")

        # Gera boxplot
        df_resultado.boxplot(column=coluna_valor, by=coluna_grupo, grid=False)
        plt.title(f"ANOVA - {coluna_valor} por {coluna_grupo}\nF={f_stat:.2f}, p={p_valor:.4f}")
        plt.suptitle("")
        plt.ylabel(coluna_valor)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("anova.png", dpi=300, bbox_inches='tight')  # salva o gráfico
        plt.show()



# -------------------------------
# Exemplo de uso
# -------------------------------
if __name__ == "__main__":
    sr = Statistic_reasoning("csv", "testes/dataset/furtoSimples")

    # Geração de tópicos
    #sr.topicos()

    # Estatísticas básicas
    pergunta = "Qual a maior Pena_Definitiva de roubo?"
    print(sr.resumo_estatisticas(1, pergunta))

    # ANOVA entre grupos
    #sr.grafico_anova(1, "Selecione Categoria furto simples e roubo e Pena_Definitiva", #coluna_grupo="Categoria")
