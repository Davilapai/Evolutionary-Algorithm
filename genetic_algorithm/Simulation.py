from Individual import Individual
import numpy as np
import random

class Simulation:
    def __init__(self, n : int, m : int):
        """
        Inits the preset values for Simulation, n is the size of the board, m is the size of the population
        """
        self.population = np.array([Individual.random_Individual_legal(n) for _ in range(m)]) 
        self.n = n
        self.m = m

    def evaluate(self):
        # Evaluate population
        evaluation = np.array([i.fitness() for i in self.population])
        
        # Sort evaluation
        evaluation = np.argsort(evaluation)

        # Apply evaluation sorting to original population
        self.population = self.population[evaluation]

        # Reverse
        self.population = self.population[::-1]
 
    def select_parents(self):
        parents = []

        # only use 2/3 of the population
        for i in range(0, int(self.m*2 / 3), 2):
            parents.append([self.population[i], self.population[i+1]])
        parents = np.array(parents)
        return parents

    @staticmethod
    def crosover(couple):
        # pmx (mantains order integrity)
        # implementation inspired by: https://ultraevolution.org/blog/pmx_crossover/
        n = len(couple[0])

        l = random.randint(0, n-1)
        r = random.choice([i for i in range(n) if i != l])

        if l > r:
            l,r = r,l

        son = [-1] * len(couple[0])
        son[l:r] = couple[0][l:r]

        for i in (*range(0,l), *range(r, len(couple[0]))):
            candidate = couple[1][i]
            while candidate in couple[0][l:r]:
                idx = np.where(couple[0] == candidate)[0][0]
                candidate = couple[1][idx]
            son[i] = candidate
        return son

    @staticmethod
    def run_simulation(n : int, m: int, runs: int, mutation_factor: float):
        """
        Runs the Simulation
        """
        
        # 1. (Initialize)
        sim = Simulation(n, m)

        for _ in range(runs):

            # 2. (Evaluate)
            sim.evaluate()
            print(f"{sim.population[0].state} fit: {sim.population[0].fitness()}")
            
            # 3. (Select Parents)
            parents = sim.select_parents()

            new_population = []
            # 4. (Crosover Parents)
            for parent1, parent2 in parents:
                new_population.append(Simulation.crosover([parent1.state, parent2.state]))
                new_population.append(Simulation.crosover([parent2.state, parent1.state]))
            for _ in range(m - len(new_population)):
                x = random.randint(0, (n-1)//2)
                y = random.randint(0, (n-1)//2)
                new_population.append(Simulation.crosover([sim.population[x].state, sim.population[y].state]))
                
            new_population = [Individual(i) for i in new_population]

            new_population = np.array(new_population)

            # 5. (Mutate)
            new_population = np.array([i.mutate(mutation_factor) for i in new_population])

            # 6. (Replace)
            sim.population = new_population

        # 7. (Result)
        print(sim.population[0].state)

Simulation.run_simulation(100, 1000, 200, 0.001)

