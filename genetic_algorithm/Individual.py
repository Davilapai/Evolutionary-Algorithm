import random
import numpy as np


class Individual:
    def __init__(self, state: list[int]):
        self.n = len(state)
        self.state = np.array(state)
        self.array = [i for i in range(self.n)]

    def fitness(self) -> int:
        """
        Returns evaluation of the current state of the individual
        """
        # dumie funcion to test
        return int(np.count_nonzero(self.array == self.state))
        """
        fit(x) = 
        """
    def mutate(self, mutation_factor: float):
        for i in range(self.n):
            if random.random() < mutation_factor:
                x = random.randint(0, self.n-1)
                self.state[i],self.state[x] = self.state[x],self.state[i]
        return self

    @staticmethod
    def random_Individual_legal(n: int):
        v = [n for n in range(n)]
        v = random.sample(v, n)
        return Individual(v)

    @staticmethod
    def random_Individual_ilegal(n: int):
        v = [random.randint(1, n) for _ in range(n)]
        return Individual(v)
