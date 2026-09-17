import random

class Engine:
    def __init__(self, n : int):
        self.n = n

    def initial_state(self):
        # Crea un arreglo desde 0 hasta n-1
        estado = list(range(self.n))
        # Lo randomiza
        random.shuffle(estado)
        # Lo retorna
        return estado

    def validate_state(self, listica : list[int]):
        # Hacemos magia muajajaja
        if set(listica) == set(range(self.n)):
            return True

        return False

    def calculate_cost(listica : list[int]):
        # Pillen como lo saco en O(n)
        # Sabemos que dada una diagona, ninguna otra reina va a poder ocuparla (CSP)
        # Creamos 2 estructuras para diagonales positivas y negativas
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

        # Si pillaron
        # Ahí los que compartan diagonal colisionan, entonces su valor va a ser > 1
        # Ahora se las sumamos al costo
        costo = 0

        for p in diag_p.values():
            if p > 1:
                # Division entera
                costo += p * (p-1) // 2

        for n in diag_n.values():
            if n > 1:
                costo += n * (n-1) //2

        return costo

    def fitness(self, listica : list[int]):
        func = self.n * (self.n - 1) // 2
        return func - self.calculate_cost(listica)
