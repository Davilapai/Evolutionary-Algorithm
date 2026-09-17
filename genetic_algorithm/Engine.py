import random

class Engine:
    def __init__(self, n : int):
        self.n = n

    def validate_state(self, listica : list[int]):
        if set(listica) == set(range(self.n)):
            return True

        return False

    @staticmethod
    def calculate_cost(listica):
        diag_p, diag_n = {},{}

        col = 0

        for fil in listica:
            suma = fil + col
            resta = fil - col
            
            # para diagonales positivas
            if suma in diag_p:
                diag_p[suma] += 1
            else:
                diag_p[suma] = 1
                
            # para diagonales negativas
            if resta in diag_n:
                diag_n[resta] += 1
            else:
                diag_n[resta] = 1
                
            col += 1

        costo = 0

        for p in diag_p.values():
            if p > 1:
                costo += p * (p-1) // 2

        for n in diag_n.values():
            if n > 1:
                costo += n * (n-1) //2

        return costo

    @staticmethod
    def fitness(listica):
        n = len(listica)
        func = n * (n - 1) // 2
        return func - Engine.calculate_cost(listica)

