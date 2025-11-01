from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import json
import numpy as np
import random
from qwen_sentencas import Qwen
from look_up_reasoning import Look_up_reasoning
from embedding_builder import Embedding_builder
from arithmetic_reasoning import Arithmetic_Reasoning
from consultaCsv import ConsultaCSV
from statistics_reasoning import Statistic_reasoning
import time
from gemini import Gemini
import matplotlib.pyplot as plt


class Reasoning_dispatcher:
    def __init__(self, epsilon=0.1, lr=0.1,
                 state_file="bandit_state.json",
                 feedback_file="feedback_buffer.json"):
        """
        modules: dicionário {nome_modulo: função}
        epsilon: taxa de exploração (quanto maior, mais aleatório)
        lr: taxa de aprendizado
        state_file: arquivo para salvar os pesos do agente
        feedback_file: arquivo para salvar feedback humano acumulado
        """

        self.modules = {
            "logical": self.logical_reasoning,
            "arithmetic": self.arithmetic_reasoning,
            "lookup": self.lookup_reasoning,
            "statistics": self.statistics_reasoning,
            "consultaRag": self.embedding_builder,
            "consultaSql": self.consulta_csv
        }

        self.epsilon = epsilon
        self.lr = lr
        self.state_file = state_file
        self.feedback_file = feedback_file
        self.callback = None
        # Inicializa embeddings
        self.embedder = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
        self.dim = len(self.embedder.embed_query("teste"))

        # Carrega estado (pesos) e feedbacks anteriores
        self.weights = {m: np.zeros(self.dim) for m in self.modules}
        if os.path.exists(self.state_file):
            self._load_state()
            print(f"[INFO] Estado carregado de '{self.state_file}'")
        else:
            print("[INFO] Novo agente iniciado (sem estado salvo).")

        # Buffer para feedback humano (memória curta)
        self.feedback_buffer = []
        if os.path.exists(self.feedback_file):
            self._load_feedback_buffer()
            print(
                f"[INFO] Buffer de feedback carregado de '{self.feedback_file}'")
        else:
            print("[INFO] Nenhum feedback humano prévio encontrado.")

     # ========= FUNÇÕES DE CALLBACK =========

    def set_callback(self, instancia_a):
        self.callback = instancia_a
    # ========= FUNÇÕES DE AGUARDE =========

    def aguarde(self, mensagem):
        aiml = Look_up_reasoning()
        response = aiml.resposta_rapida(mensagem)
        return response

    # ========= FUNÇÕES DE EMBEDDING E ESCOLHA =========
    def get_embedding(self, text):
        # msg = self.aguarde("espera")
        # self.callback.send_mensagem(self.id, msg)
        # emb = Embedding_builder()
        # resp = emb.run_graph(mensagem)
        # response = resp['answer']
        """Gera o embedding do texto usando o modelo configurado"""
        return np.array(self.embedder.embed_query(text))

    def select_module(self, embedding):
        """
        Escolhe um módulo:
        - Com probabilidade epsilon: escolha aleatória (exploração)
        - Caso contrário: escolha com maior produto escalar (exploração/exploração)
        """
        if random.random() < self.epsilon:
            return random.choice(list(self.modules.keys()))
        scores = {m: np.dot(embedding, self.weights[m]) for m in self.modules}
        return max(scores, key=scores.get)

    # ========= FUNÇÕES DE ATUALIZAÇÃO =========
    def update(self, module, embedding, reward):
        """
        Atualiza os pesos do módulo escolhido com base na recompensa recebida.
        reward: número (ex: 1 para acerto, 0 para erro)
        """
        prediction = np.dot(embedding, self.weights[module])
        self.weights[module] += self.lr * (reward - prediction) * embedding
        self._save_state()

    # ========= PROCESSAMENTO =========
    def process_text(self, id, mensagem):
        # Gera embedding do texto
        embedding = self.get_embedding(mensagem)
        print("embedding")
        # Seleciona o módulo mais apropriado
        module_name = self.select_module(embedding)
        print(module_name)
        # Executa o módulo passando id e mensagem
        result = self.modules[module_name](id, mensagem)

        return module_name, result, embedding
    
    # ========= PROCESSAMENTO TREINO =========
    def process_text_train(self, id, mensagem):
        # Gera embedding do texto
        embedding = self.get_embedding(mensagem)

        # Seleciona o módulo mais apropriado
        module_name = self.select_module(embedding)

        # Não executa o módulo real, apenas retorna um "dummy"
        result = None  # ou até uma string tipo "dummy result"

        return module_name, result, embedding


    # ========= FEEDBACK HUMANO =========

    def human_feedback(self, text, chosen_module, reward):
        """
        Recebe um texto, o módulo que processou e a recompensa humana.
        - Atualiza imediatamente os pesos
        - Armazena no buffer para treino posterior
        """
        embedding = self.get_embedding(text)
        self.update(chosen_module, embedding, reward)
        self.feedback_buffer.append((text, chosen_module, reward))
        self._save_feedback_buffer()
        print(
            f"[HUMAN FEEDBACK] Módulo '{chosen_module}' atualizado com recompensa {reward}.")

    # ========= TREINAMENTO AUTOMÁTICO =========
    def auto_train(self, training_data):
        """
        Treina automaticamente a partir de um dataset com módulo definido no JSON.
        training_data: lista de dicts {"text": ..., "module": ..., "reward": ...}
        """
        for item in training_data:
            if isinstance(item, dict):  # formato JSON/dict
                text = item["text"]
                chosen_module = item["module"]
                reward = item["reward"]
            else:  # formato tupla/lista
                text, chosen_module, reward = item

            embedding = self.get_embedding(text)
            self.update(chosen_module, embedding, reward)

            print(
                f"[AUTO TRAIN] '{text}' -> módulo '{chosen_module}' atualizado com recompensa {reward}"
            )
    # ========= TREINO A PARTIR DE FEEDBACK HUMANO =========

    def train_from_feedback(self, clear_buffer=True):
        """
        Treina usando todos os feedbacks humanos armazenados no buffer.
        clear_buffer: se True, apaga o buffer após o treino.
        """
        if not self.feedback_buffer:
            print("[INFO] Nenhum feedback humano armazenado.")
            return
        print(
            f"[INFO] Treinando a partir de {len(self.feedback_buffer)} feedback(s) humano(s)...")
        self.auto_train(self.feedback_buffer)
        if clear_buffer:
            self.feedback_buffer.clear()
            self._save_feedback_buffer()

    # ========= SALVAR E CARREGAR =========
    def _save_state(self):
        """Salva os pesos atuais no arquivo JSON"""
        state_dict = {m: self.weights[m].tolist() for m in self.weights}
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, ensure_ascii=False, indent=2)

    def _load_state(self):
        """Carrega pesos salvos anteriormente"""
        with open(self.state_file, "r", encoding="utf-8") as f:
            state_dict = json.load(f)
        self.weights = {m: np.array(state_dict[m]) for m in state_dict}

    def _save_feedback_buffer(self):
        """Salva o buffer de feedback humano em arquivo"""
        with open(self.feedback_file, "w", encoding="utf-8") as f:
            json.dump(self.feedback_buffer, f, ensure_ascii=False, indent=2)

    def _load_feedback_buffer(self):
        """Carrega o buffer de feedback humano de arquivo"""
        with open(self.feedback_file, "r", encoding="utf-8") as f:
            self.feedback_buffer = json.load(f)

    # ======== MÓDULOS DE TREINAMENTO ========

    def embedding_builder(self, id, mensagem):
        self.id = id
        msg = self.aguarde("espera")
        # print(msg)
        self.callback.send_mensagem(self.id, msg)
        emb = Embedding_builder()
        resp = emb.run_graph(mensagem)
        response = resp['answer']
        return response
        #return "rr"
    def lookup_reasoning(self, id, mensagem):
        self.id = id
        aiml = Look_up_reasoning()
        response = aiml.resposta_rapida(mensagem)
        return response

    def logical_reasoning(self, id, mensagem):
        self.id = id
        return f"[Logical] Inferindo regras sobre: {mensagem}"

    def arithmetic_reasoning(self, id, mensagem):
        self.id = id
        self.arithmetic = Arithmetic_Reasoning()
        try:
            resultado = self.arithmetic.evaluate(mensagem)
            return f"[Arithmetic] Resultado: {resultado}"
        except Exception as e:
            return f"[Arithmetic] Não foi possível calcular ({e})"

    def statistics_reasoning(self, id, mensagem):
        self.id = id
        print("modulo estatistico")
        msg = self.aguarde("espera")
        # print(msg)
        self.callback.send_mensagem(self.id, msg)

        sr = Statistic_reasoning("csv", "testes/dataset/furtosimples")

        mensagem_normalizada = mensagem.lower()  # transforma em minusculo
        

        if "grafico" in mensagem_normalizada:
            resultado = sr.grafico_anova(id, mensagem, coluna_grupo="categoria")
        else:
            resultado = sr.resumo_estatisticas(id, mensagem)

        return resultado
            


    def consulta_csv(self, id, mensagem):
        self.id = id
        print("modulo consulta")
        msg = self.aguarde("espera")
        # print(msg)
        self.callback.send_mensagem(self.id, msg)
        consulta = ConsultaCSV("csv")
        resultado = consulta.respostaLlm(id, mensagem)
        print(resultado)
        if ((isinstance(resultado, str) and resultado == "None") or resultado.empty):
            resp = self.embedding_fall(id, mensagem)
        else:
            print("ttt")
            resp = consulta.verificaResp(mensagem,resultado)

        return resp

    def embedding_fall(self, id, mensagem):
        self.id = id
        print("modulo embedding fall")
        emb = Embedding_builder()
        resp = emb.run_graph(mensagem)
        response = resp['answer']
        return response
    
    def consulta_fall(self, id, mensagem):
        self.id = id
        print("modulo consulta fall")
        consulta = ConsultaCSV("csv")
        resultado = consulta.respostaLlm(id, mensagem)
        resp = consulta.verificaResp(mensagem,resultado) #possivelmente vai mandar que nao ha dados
        return resp

    def respostaLlm(self, id, mensagem):
        self.id = id
        qwen = Qwen()
        response = qwen.enviar_msg(mensagem)
        return response


if __name__ == "__main__":
    """
    #======== Treinamento com IA========
    contador = 0
    tempo_inicial = time.time()
    window = 50  # tamanho da janela para média móvel
    agent = Reasoning_dispatcher(epsilon=0.2, lr=0.2)
    gemini = Gemini()
    qwen = Qwen()
        
    # Logs de treinamento
    acertos = []
    escolhas = {m: 0 for m in agent.modules}   # contagem por módulo
    historico_normas = {m: [] for m in agent.modules}

    while contador < 500:
        contador+=1
        
        contexto_pergunta, categoria = gemini.build_random_prompt()
        #pergunta = gemini.enviar_msg(contexto_pergunta)
        pergunta = qwen.enviar_msg(contexto_pergunta)
        # Processa texto e pede feedback humano
    
        modulo, saida, _ = agent.process_text_train("0", pergunta)
        

        recompensa = 0

        if categoria == modulo:
            recompensa = 1

        print(f"Entrada: {pergunta} | Módulo: {modulo} | Saída: {saida} | Reconpensa: {recompensa}")
        agent.human_feedback(pergunta, modulo, recompensa)

        # Treino posterior com feedback acumulado
        #
        agent.train_from_feedback()

        if contador %10 == 0:
            tempoEspera = 80 - (time.time() - tempo_inicial)
            print(tempoEspera)
            if tempoEspera > 0:
                time.sleep(tempoEspera)
            #contador = 0
            tempo_inicial = time.time()
    # Guarda métricas
        acertos.append(recompensa)
        escolhas[modulo] += 1
        for m in agent.modules:
            historico_normas[m].append(np.linalg.norm(agent.weights[m]))
    """
    # ======== EXEMPLO DE USO manual========
    while 1:
        agent = Reasoning_dispatcher(epsilon=0.2, lr=0.2)
        #gemini = Gemini()
        # Processa texto e pede feedback humano
        texto = input("Digite uma pergunta: ")
        modulo, saida, _ = agent.process_text_train("12", texto)
        print(f"Entrada: {texto} | Módulo: {modulo} | Saída: {saida}")
        recompensa = int(input("Avalie a resposta (1=certo, 0=errado): "))
        agent.human_feedback(texto, modulo, recompensa)

        # Treino posterior com feedback acumulado
        #
        agent.train_from_feedback()
    
    """
    # ======== Treinamento automatico ========
    agent = Reasoning_dispatcher(epsilon=0.2, lr=0.2)
    # ======================
    # Dicionário de perguntas por módulo
    # ======================
    perguntas_por_modulo = {
        "arithmetic": [
            "2 + 2",
            "quanto é 2 vezes 3",
            "3 - 1",
            "10 * 5",
            "100 / 4",
            "5 ** 2",
            "sqrt(16)",
            "7 + 8 - 3",
            "9 * 9",
            "50 / 5",
            "2 ** 8",
            "raiz quadrada de 81",
            "quanto é 10 mais 20?",
            "subtraia 15 de 100",
            "multiplique 12 por 3",
            "divida 144 por 12",
            "qual o resto de 10 dividido por 3?",
            "5 + 3 * 2",
            "(5 + 3) * 2",
            "10 - 4 + 2",
            "100 - (25 * 2)",
            "2 elevado a 10",
            "log de 100",
            "log10(1000)",
            "quanto é 1/4 de 100?",
            "qual a porcentagem de 20 em 80?",
            "40% de 200 é quanto?",
            "qual é a média de 10, 20 e 30?",
            "some 7, 8 e 9",
            "faça 3 vezes 5 mais 2",
        ],

        "logical": [
            "se todos A são B e todos B são C, todos A são C?",
            "se Maria é mais velha que João e João é mais velho que Ana, quem é mais velho?",
            "se chove, a rua molha; está chovendo, a rua está molhada?",
            "todos os cães são mamíferos, Rex é um cão, Rex é mamífero?",
            "se todos os pássaros voam e um pinguim é um pássaro, o pinguim voa?",
            "se hoje é segunda e amanhã é depois de segunda, que dia é amanhã?",
            "se a lâmpada está desligada, há luz?",
            "todo quadrado é um retângulo?",
            "se um número é par, ele é divisível por 2?",
            "se A implica B e B implica C, então A implica C?",
            "João é mais alto que Pedro e Pedro é mais alto que Lucas, quem é o mais baixo?",
            "se todos os juízes são humanos e João é juiz, João é humano?",
            "se o réu confessou, ele é culpado?",
            "é possível que alguém seja inocente e culpado ao mesmo tempo?",
            "um triângulo pode ter quatro lados?",
            "se X é maior que Y e Y é maior que Z, X é maior que Z?",
            "se um processo tem reincidência, ele também tem antecedentes?",
            "se não há provas, há condenação?",
            "se todo crime tem pena e este ato é crime, ele tem pena?",
            "toda sentença é decisão?",
            "pode algo ser verdadeiro e falso ao mesmo tempo?",
            "se algo é necessário, é também possível?",
            "se um réu é absolvido, ele é considerado culpado?",
            "é possível que 2 seja maior que 3?",
            "se A é igual a B e B é igual a C, A é igual a C?",
            "se uma premissa é falsa, a conclusão é verdadeira?",
            "é verdadeiro que todo homem é mortal?",
            "se choveu ontem e hoje o chão está molhado, é correto inferir causalidade?",
            "a negação de uma negação retorna o valor original?",
            "se uma condição é suficiente, ela é também necessária?",
        ],

        "statistics": [
            "Qual é a média da Pena Base para cada categoria de crime?",
            "Qual é a mediana da Pena Definitiva para processos com reincidência?",
            "Qual é o desvio padrão da Pena Base nas varas de determinada comarca?",
            "Qual foi o valor mínimo da Pena Definitiva em casos de Furto Simples?",
            "Qual foi o valor máximo da Pena Definitiva em casos de Furto Simples?",
            "Quantos processos têm registro de Pena Base para cada categoria?",
            "Qual é a variância da Pena Base por categoria?",
            "Qual é a média de antecedentes por comarca?",
            "Qual é o total de processos analisados?",
            "Qual é o intervalo interquartil da Pena Definitiva?",
            "Qual é o desvio absoluto médio da Pena Base?",
            "Quantos processos possuem reincidência?",
            "Qual a moda das penas aplicadas em casos de roubo?",
            "Qual é a média de conversões de pena por categoria?",
            "Qual é o valor mínimo da Pena Base mínimo por comarca?",
            "Qual é o valor máximo da Pena Base mínimo por comarca?",
            "Qual é o número médio de antecedentes por categoria?",
            "Qual categoria apresenta maior média de Pena Definitiva?",
            "Qual categoria possui menor desvio padrão na Pena Base?",
            "Quantos processos possuem confissão espontânea?",
            "Qual é a média de penas para crimes de tráfico de drogas?",
            "Quantos processos tiveram pena convertida?",
            "Qual é a proporção de reincidência por categoria?",
            "Qual é a correlação entre antecedentes e pena definitiva?",
            "Qual é a distribuição de penas por tipo de crime?",
            "Qual a variância da Pena Definitiva em furto qualificado?",
            "Qual é o valor médio da conversão de pena por comarca?",
            "Qual é o número de réus reincidentes por vara?",
            "Qual é o desvio padrão da pena base em casos de roubo?",
            "Qual categoria possui mais registros de reincidência?",
        ],

        "lookup": [
            "bom dia", "boa noite", "olá", "oi", "tchau", "até logo",
            "não entendi", "não gostei da resposta", "resposta insuficiente",
            "obrigado", "tudo bem?", "como vai?", "agradeço pela ajuda",
            "pode repetir?", "pode explicar melhor?", "não faz sentido",
            "por favor", "entendi", "entendi agora", "certo", "claro",
            "pode continuar", "vamos prosseguir", "ótimo", "excelente resposta",
            "essa explicação está errada", "resposta incompleta",
            "não respondeu o que perguntei", "explique de outro jeito",
            "não compreendi sua resposta",
        ],

        "consultaRag": [
            "como é definida a pena para furto simples?",
            "quais circunstâncias podem agravar um furto qualificado?",
            "quais fatores diferenciam furto simples de furto qualificado?",
            "quais elementos caracterizam o tráfico de drogas?",
            "em quais situações o tráfico pode ter pena reduzida?",
            "quais agravantes podem aumentar a pena de roubo?",
            "como diferenciar roubo de furto na prática?",
            "quais são as consequências jurídicas do furto qualificado?",
            "quais são as penas previstas para o crime de roubo?",
            "quando o furto pode ser considerado privilegiado?",
            "quais critérios o juiz considera na dosimetria da pena?",
            "como é calculada a pena definitiva?",
            "em que casos há substituição da pena por restritiva de direitos?",
            "o que caracteriza reincidência no direito penal?",
            "o que significa confissão espontânea?",
            "como são analisadas as circunstâncias judiciais?",
            "quais fatores podem reduzir a pena base?",
            "como é aplicada a atenuante da menoridade relativa?",
            "em que situações ocorre agravante de reincidência?",
            "quais elementos compõem a pena definitiva?",
            "o que diferencia a pena base da pena intermediária?",
            "quando há possibilidade de suspensão condicional da pena?",
            "como é definida a pena mínima em cada tipo penal?",
            "qual a diferença entre tentativa e consumação no cálculo da pena?",
            "quais são as causas de aumento de pena no tráfico?",
            "como funciona a continuidade delitiva?",
            "em quais casos é possível regime semiaberto?",
            "como o juiz define o regime inicial da pena?",
            "o que é concurso material de crimes?",
            "como ocorre a unificação de penas?",
        ],

        "consultaSql": [
            "qual a categoria do processo 0000001-69.2013.8.26.0500?",
            "quem é o juiz do processo 0000002-35.2014.8.26.0001?",
            "qual a comarca do processo 0000003-55.2015.8.30.0538?",
            "qual a vara responsável pelo processo 0000004-00.2016.8.26.0538?",
            "qual a pena base do processo 0000005-78.2017.8.99.0538?",
            "qual a pena definitiva do processo 0000006-41.2025.8.26.0538?",
            "o réu do processo 0000007-90.2019.8.26.0999 tem antecedentes?",
            "o processo 0000008-22.2020.8.26.0538 teve reincidência?",
            "o réu do processo 0000009-11.2021.8.26.0538 confessou espontaneamente?",
            "qual o valor mínimo da pena base no processo 0000010-22.2022.8.26.0538?",
            "houve conversão da pena no processo 0000011-77.2023.8.26.0538?",
            "qual é a pena intermediária do processo 0000012-00.2019.8.26.0538?",
            "qual foi a decisão final do processo 0000013-09.2018.8.26.0538?",
            "quem é o réu do processo 0000014-55.2020.8.26.0538?",
            "qual é a data da sentença do processo 0000015-88.2021.8.26.0538?",
            "qual o tipo de crime no processo 0000016-33.2019.8.26.0538?",
            "houve confissão no processo 0000017-66.2020.8.26.0538?",
            "qual foi o regime inicial da pena no processo 0000018-99.2019.8.26.0538?",
            "qual é o número de antecedentes no processo 0000019-77.2021.8.26.0538?",
            "quem foi o defensor no processo 0000020-11.2022.8.26.0538?",
            "qual é a comarca do processo 0000021-55.2017.8.26.0538?",
            "qual foi o resultado final do processo 0000022-66.2018.8.26.0538?",
            "houve recurso no processo 0000023-00.2016.8.26.0538?",
            "qual o valor máximo da pena base no processo 0000024-12.2020.8.26.0538?",
            "quem é o magistrado responsável pelo processo 0000025-10.2022.8.26.0538?",
            "qual é o status atual do processo 0000026-98.2023.8.26.0538?",
            "qual a vara criminal do processo 0000027-45.2015.8.26.0538?",
            "qual é o crime principal do processo 0000028-31.2020.8.26.0538?",
            "quem é o autor da ação no processo 0000029-77.2022.8.26.0538?",
            "o processo 0000030-88.2023.8.26.0538 foi arquivado?",
        ],
    }

    # Junta todas as perguntas para sorteio aleatório
    todas_entradas = [p for perguntas in perguntas_por_modulo.values() for p in perguntas]

    # ======================
    # Simulação de treinamento
    # ======================
    steps = 2000
    window = 100

    acertos = []
    escolhas = {m: 0 for m in perguntas_por_modulo.keys()}
    historico_normas = {m: [] for m in perguntas_por_modulo.keys()}

    for step in range(steps):
        texto = random.choice(todas_entradas)
        mod, output, emb = agent.process_text_train("id", texto)

        # Verifica o módulo correto comparando a entrada exata
        modulo_correto = None
        for nome_modulo, perguntas in perguntas_por_modulo.items():
            if texto in perguntas:
                modulo_correto = nome_modulo
                break

        reward = 1 if mod == modulo_correto else 0
        print(f"Entrada: {texto} | Módulo: {mod} | Esperado: {modulo_correto} | Recompensa: {reward}")

        # Atualiza o agente
        agent.update(mod, emb, reward)

        # Guarda métricas
        acertos.append(reward)
        escolhas[mod] += 1
        for m in perguntas_por_modulo.keys():
            historico_normas[m].append(np.linalg.norm(agent.weights[m]))

    # -----------------------------
    # GRÁFICOS
    # -----------------------------

    # 1. Convergência (rolling accuracy)
    rolling_acc = np.convolve(acertos, np.ones(window)/window, mode="valid")

    plt.figure(figsize=(10,5))
    plt.plot(rolling_acc, label=f"Acurácia média (janela={window})", color="blue")
    plt.axhline(y=1.0, color="green", linestyle="--", label="Convergência ideal")
    plt.xlabel("Iteração")
    plt.ylabel("Acurácia (rolling)")
    plt.title("Convergência da taxa de acerto")
    plt.legend()
    plt.grid()
    plt.savefig("convergencia_acuracia.png", dpi=300)
    plt.show()

    # 2. Normas dos pesos ao longo do tempo
    plt.figure(figsize=(10,6))
    for m, valores in historico_normas.items():
        plt.plot(valores, label=m)
    plt.xlabel("Iteração")
    plt.ylabel("Norma dos pesos")
    plt.title("Evolução das normas dos vetores de pesos")
    # Legenda fixa no canto superior esquerdo
    plt.legend(loc="upper left")
    
    plt.grid()
    plt.savefig("evolucao_pesos.png", dpi=300)
    plt.show()

    # 3. Distribuição de escolhas dos módulos
    plt.figure(figsize=(8,5))
    plt.bar(escolhas.keys(), escolhas.values())
    plt.xlabel("Módulo")
    plt.ylabel("Quantidade de escolhas")
    plt.title("Distribuição de escolhas de módulos")
    plt.savefig("distribuicao_escolhas.png", dpi=300)
    plt.show()"""
