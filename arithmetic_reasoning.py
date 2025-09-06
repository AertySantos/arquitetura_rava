import ast
import operator

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

    def evaluate(self, expr: str):
        expr = self._preprocess(expr)
        tree = ast.parse(expr, mode="eval")
        return self._eval_node(tree.body)

