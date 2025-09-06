import pandas as pd
from qwen_sentencas import Qwen
import duckdb
import re
import unidecode
import glob
import os

class ConsultaCSV:
    def __init__(self, pasta_csv):
        # Localiza todos os arquivos CSV na pasta
        arquivos = glob.glob(os.path.join(pasta_csv, "*.csv"))

        if not arquivos:
            raise FileNotFoundError(f"Nenhum arquivo CSV encontrado em {pasta_csv}")

        # Colunas comuns que queremos manter
        colunas_comuns = {
            "Número processo": "Numero_processo",
            "Categoria": "Categoria",
            "comarca": "Comarca",
            "vara": "Vara",
            "juiz": "Juiz",
            "Pena Base": "Pena_Base",
            "Pena Definitiva": "Pena_Definitiva",
            "Antecedentes": "Antecedentes",
            "Pena-Base Minimo?": "Pena_Base_Minimo",
            "Reincidência": "Reincidencia",
            "Confissão Espontânea": "Confissao_Espontanea",
            "Conversão da Pena": "Conversao_da_Pena",
        }

        # Lista para armazenar todos os DataFrames
        dataframes = []

        for arquivo in arquivos:
            df = pd.read_csv(arquivo, sep=",", on_bad_lines="skip")

            # Renomeia apenas as colunas comuns
            df.rename(columns=colunas_comuns, inplace=True)

            # Mantém somente as colunas comuns, na ordem definida
            df = df[list(colunas_comuns.values())]

            # Normaliza nomes (remove espaços extras e acentos)
            df.columns = [unidecode.unidecode(col.strip()) for col in df.columns]

            dataframes.append(df)

        # Concatena todos os CSVs em um único DataFrame
        self.df = pd.concat(dataframes, ignore_index=True)

        #print(self.df)

    def pergunta_para_sql(self, pergunta: str) -> str:
        """
        Converte a pergunta natural em SQL usando o LLM.
        """
        prompt = f"""
        Você deve transformar a pergunta do usuário em uma consulta SQL válida 
        para ser executada sobre um DataFrame chamado df (carregado a partir de um CSV).  

        O DataFrame contém as seguintes colunas:  
            - Numero_processo  
            - Categoria  
            - Comarca  
            - Vara  
            - Juiz  
            - Pena_Base  
            - Pena_Definitiva  
            - Antecedentes  
            - Pena_Base_Minimo  
            - Reincidencia  
            - Confissao_Espontanea  
            - Conversao_da_Pena  
    
        ⚠️ Regras:  
        1. Use sempre **nomes exatos das colunas acima**.  
        2. Sempre consulte a tabela `df`.  
        3. Retorne **apenas o SQL puro**, sem explicações, sem markdown, sem ```sql.  
        
        Pergunta do usuário: "{pergunta}"  
        SQL:
        """
        qwen = Qwen()
        sql_query = qwen.enviar_msg(prompt)
        return sql_query.strip()

    def executar_sql(self, sql_query: str) -> pd.DataFrame:
        """
        Executa a consulta SQL no DataFrame com DuckDB.
        """
        try:
            # Registra o DataFrame como tabela 'df'
            duckdb.register("df", self.df)

            # Executa a consulta
            resultado = duckdb.query(sql_query).to_df()
            return resultado
        except Exception as e:
            print("Erro ao executar SQL:", e)
            return pd.DataFrame()

    def respostaLlm(self, id, mensagem):
        """
        Função principal: transforma pergunta em SQL e retorna resultado.
        """
        sql_query = self.pergunta_para_sql(mensagem)
        #sql_query = "SELECT Pena_Definitiva FROM df WHERE Numero_processo = '0014485-64.2020.8.26.0564'"
        # Remove possíveis marcadores do LLM
        sql_query = re.sub(r"```sql|```|::", "", sql_query).strip()
        #print
        print(f"{sql_query}")  # debug
        resultado = self.executar_sql(sql_query)
        return resultado

    def verificaResp(self, pergunta: str, resp: str) -> str:
        """
        Verifica a resposta obtida. 
        Se resp estiver vazio/NaN → consulta o LLM para verificar se a pergunta contém dados suficientes.
        Se resp existir → consulta o LLM para formatar a resposta em português formal.
        """

        # Prompt comum para todos os casos
        contexto = """
        O DataFrame contém as seguintes colunas:  
         - Numero_processo  
            - Categoria  
            - Comarca  
            - Vara  
            - Juiz  
            - Pena_Base  
            - Pena_Definitiva  
            - Antecedentes  
            - Pena_Base_Minimo  
            - Reincidencia  
            - Confissao_Espontanea  
            - Conversao_da_Pena    

        ⚠️ Regras:  
        1. Sempre use os nomes exatos das colunas acima.  
        """

        if resp is None or str(resp).lower() in ["nan", "none", ""]:
            # Prompt para quando não há resposta
            prompt = f"""
            {contexto}

            A resposta retornou vazia ou NaN.  
            Pergunta do usuário: "{pergunta}"  

            Tarefa:  
            - Verifique se a pergunta contém informações suficientes para extrair a resposta das colunas disponíveis.  
            - Caso falte alguma informação essencial, peça educadamente ao usuário para fornecê-la, sem termos tecnicos (dataframe, colunas...).  
            """
        else:
            # Prompt para quando já existe uma resposta
            prompt = f"""
            {contexto}

            Pergunta do usuário: "{pergunta}"  
            Resposta extraída: "{resp}"  

             Tarefa:  
            - Reformule a resposta acima em português formal.
            - A resposta deve ser clara, direta e bem estruturada, sem termos técnicos (dataframe, colunas...).  
            """

        qwen = Qwen()
        sql_query = qwen.enviar_msg(prompt)
        return sql_query.strip()


    def costuraLlm(self, id, mensagem, resp):
       
        #sql_query = self.pergunta_para_sql(mensagem)
        #sql_query = "SELECT Pena_Definitiva FROM df WHERE Numero_processo = '0014485-64.2020.8.26.0564'"
        # Remove possíveis marcadores do LLM
        sql_query = re.sub(r"```sql|```|::", "", sql_query).strip()
        #print
        print(f"{sql_query}")  # debug
        resultado = self.executar_sql(sql_query)
        return resultado

# Exemplo de uso
if __name__ == "__main__":
    consulta = ConsultaCSV("csv")
    pergunta = "Qual o tipo de crime do processo de numero 0000001-69.2013.8.26.0538?"
    resultado = consulta.respostaLlm(1, pergunta)
    costura = consulta.verificaResp(pergunta,resultado)
    print(costura)

