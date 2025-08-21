from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import json
import numpy as np
import random
from qwen_sentencas import Qwen
from look_up_reasoning import Look_up_reasoning
from embedding_builder import Embedding_builder


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
            "embedding": self.embedding_builder
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

        # Seleciona o módulo mais apropriado
        module_name = self.select_module(embedding)

        # Executa o módulo passando id e mensagem
        result = self.modules[module_name](id, mensagem)

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

    def lookup_reasoning(self, id, mensagem):
        self.id = id
        aiml = Look_up_reasoning()
        response = aiml.resposta_rapida(mensagem)
        return response

    def logical_reasoning(self, id, mensagem):
        return f"[Logical] Inferindo regras sobre: {mensagem}"

    def arithmetic_reasoning(self, id, mensagem):
        try:
            return f"[Arithmetic] Resultado: {eval(mensagem)}"
        except:
            return "[Arithmetic] Não foi possível calcular"

    def statistics_reasoning(self, id, mensagem):
        return f"[Statistics] Padrão detectado no texto '{mensagem}': Grupo {hash(mensagem) % 3}"

    def respostaLlm(self, id, mensagem):

        qwen = Qwen()
        response = qwen.enviar_msg(mensagem)
        return response


if __name__ == "__main__":

    # ======== EXEMPLO DE USO ========
    agent = Reasoning_dispatcher(epsilon=0.2, lr=0.2)
    # Processa texto e pede feedback humano
    texto = "12 + 10"
    modulo, saida, _ = agent.process_text("12", texto)
    print(f"Entrada: {texto} | Módulo: {modulo} | Saída: {saida}")
    recompensa = int(input("Avalie a resposta (1=certo, 0=errado): "))
    agent.human_feedback(texto, modulo, recompensa)

    # Treino posterior com feedback acumulado
    #
    agent.train_from_feedback()
    """
    
    # ======== Simulação ========
    agent = Reasoning_dispatcher(epsilon=0.2, lr=0.2)
    entradas = [
        # Aritmético
        "2 + 2",
        "3 - 1",
        "10 * 5",
        "100 / 4",
        "5 ** 2",
        "sqrt(16)",
        "7 + 8 - 3",

        # Lógico
        "se todos A são B e todos B são C, todos A são C?",
        "se Maria é mais velha que João e João é mais velho que Ana, quem é mais velho?",
        "se chove, a rua molha; está chovendo, a rua está molhada?",
        "todos os cães são mamíferos, Rex é um cão, Rex é mamífero?",

        # Estatístico
        "média",
        "mediana",
        "moda",
        "variância",
        "desvio padrão",
        "analisar dispersão",
        "detectar tendência",

        # Lookup
        "bom dia",
        "boa noite",
        "olá",
        "oi",
        "tchau",
        "até logo",

        # Embedding builder
        "qual a sentença de fulano?",
        "qual a pena-base de roubo?",
        "qual a pena por furto?",
        "pena de homicídio",
        "tempo de prisão por corrupção"
    ]

    # Palavras-chave para cada módulo
    keywords = {
        "arithmetic": ["+", "-", "*", "/", "**", "sqrt"],
        "logical": ["A são B", "mais velho", "se chove", "todos os cães"],
        "statistics": ["média", "mediana", "moda", "variância", "desvio padrão", "dispersão", "tendência"],
        "lookup": ["bom dia", "boa noite", "olá", "oi", "tchau", "até logo"],
        "embedding": ["sentença", "pena-base", "pena", "prisão", "homicídio", "corrupção"]
    }

    for step in range(300):
        texto = random.choice(entradas)
        mod, output, emb = agent.process_text(texto)
        print(
            f"Passo {step} | Entrada: '{texto}' | Módulo: {mod} | Saída: {output}")

        # Determinar recompensa com base nas palavras-chave
        reward = 0
        for module_name, words in keywords.items():
            if any(word.lower() in texto.lower() for word in words):
                reward = 1 if mod == module_name else 0
                break

        agent.update(mod, emb, reward)

    print("\nNorma dos vetores de pesos aprendidos:")
    for m in agent.modules:
        print(f"{m}: {np.linalg.norm(agent.weights[m]):.2f}")"""
