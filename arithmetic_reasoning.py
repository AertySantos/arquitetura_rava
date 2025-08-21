import math


class ArithmeticReasoning:
    def __init__(self):
        pass

    # Operações básicas
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        if b == 0:
            return "Erro: divisão por zero"
        return a / b

    # Áreas
    def area_square(self, side):
        return side ** 2

    def area_rectangle(self, length, width):
        return length * width

    def area_circle(self, radius):
        return math.pi * (radius ** 2)


# Exemplo de uso
calc = ArithmeticReasoning()

print("Soma:", calc.add(5, 3))
print("Divisão:", calc.divide(10, 2))
print("Área do quadrado:", calc.area_square(4))
print("Área do círculo:", calc.area_circle(3))
