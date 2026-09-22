import random
import numpy as np

from .Engine import Engine

class Individual:
    def __init__(self, state: list[int]):
        self.n = len(state)
        self.state = np.array(state)

    def fitness(self) -> int:
        """
        Returns evaluation of the current state of the individual
        """
        # dumie funcion to test
        return Engine.fitness(self.state)

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
