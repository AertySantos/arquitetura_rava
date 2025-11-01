import ast
import operator
from qwen_sentencas import Qwen

class Arithmetic_Reasoning:

    # operadores permitidos
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
    }

    # substituições de texto
    replacements = {
        "x": "*",
        "X": "*",
        "vezes": "*",
        "÷": "/",
        ":": "/",
        "dividido por": "/",
        "mais": "+",
        "menos": "-",
        "elevado a": "**",
    }

    def _preprocess(self, expr: str) -> str:
        expr = expr.lower()
        for k, v in self.replacements.items():
            expr = expr.replace(k, v)
        return expr

    def _eval_node(self, node):
        if isinstance(node, ast.Num):  # Python <3.8
            return node.n
        elif isinstance(node, ast.Constant):  # Python 3.8+
            return node.value
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op_type = type(node.op)
            if op_type in self.operators:
                return self.operators[op_type](left, right)
            else:
                raise ValueError("Operador não permitido")
        else:
            raise ValueError("Expressão inválida")

    def mathonly(self, pergunta: str) -> str:
        """
        Converte uma pergunta em linguagem natural para uma expressão matemática.
        """
        prompt = f"""
        Você é um assistente que converte perguntas em linguagem natural sobre cálculos
        em expressões matemáticas válidas no formato Python.

        ⚙️ Instruções:
        - Entenda o significado da pergunta.
        - Retorne **apenas** a expressão matemática equivalente.
        - Não explique, não adicione texto, nem unidades de medida.
        - Use operadores matemáticos padrão (+, -, *, /, **, sqrt, etc.).
        - Se a pergunta não for matemática, retorne "None".

        Exemplos:
        - "quanto é 2 mais 2" → 2 + 2  
        - "qual é a raiz quadrada de 16" → sqrt(16)  
        - "multiplique 5 por 8" → 5 * 8  
        - "dez dividido por dois" → 10 / 2  
        - "qual é o valor de pi" → pi  
        - "qual é o seu nome?" → None  

        Pergunta do usuário: "{pergunta}"
        Responda apenas com a expressão:
        """
        qwen = Qwen()
        resposta = qwen.enviar_msg(prompt).strip()
        return resposta

    def evaluate(self, pergunta: str):
        """
        Recebe uma pergunta em linguagem natural ou expressão matemática
        e retorna o resultado. Tenta primeiro avaliar diretamente;
        se falhar, usa o LLM para converter a pergunta em expressão.
        """
        expr_original = pergunta.strip()
        expr = self._preprocess(expr_original)

        # Tenta avaliar diretamente primeiro
        try:
            tree = ast.parse(expr, mode="eval")
            resultado = self._eval_node(tree.body)
            return resultado
        except Exception:
            # Se não conseguir avaliar, tenta converter com o LLM
            expr_llm = self.mathonly(expr_original)

            if not expr_llm or expr_llm.lower() == "none":
                return "Não é uma pergunta matemática."

            expr_llm = self._preprocess(expr_llm)
            try:
                tree = ast.parse(expr_llm, mode="eval")
                resultado = self._eval_node(tree.body)
                return resultado
            except Exception as e:
                return f"Erro ao avaliar expressão gerada: {e}"

