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
        fitness = [individual.fitness() for individual in self.population]
        self.population = self.population[np.argsort(fitness)[::-1]]
 
    def select_parents(self):
        parents = []
        for i in range(0, 2 * self.m // 3 - 1, 2):
            parents.append([self.population[i], self.population[i + 1]])
        return np.array(parents)

    @staticmethod
    def crosover(couple):
        # pmx (mantains order integrity)
        # implementation inspired by: https://ultraevolution.org/blog/pmx_crossover/
        p1 = couple[0]
        p2 = couple[1]
        n = len(p1)

        l = random.randint(0, n-1)
        r = random.choice([i for i in range(1, n+1) if i != l])

        if l > r:
            l,r = r,l

        son = np.array([-1] * n)
        son[l:r] = p1[l:r]

        for i in (*range(0, l), *range(r, n)):
            candidate = p2[i]
            while candidate in p1[l:r]:
                idx = list(p1).index(candidate)
                candidate = p2[idx]
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
            for _ in range(m - 2 - len(new_population)):
                x = random.randint(0, (m-1)//2)
                y = random.randint(0, (m-1)//2)
                new_population.append(Simulation.crosover([sim.population[x].state, sim.population[y].state]))
                
            new_population = [Individual(i) for i in new_population]
            
            # 5. (Mutate)
            new_population = [individual.mutate(mutation_factor) for individual in new_population]

            # Elitism
            new_population.append(sim.population[0])
            new_population.append(sim.population[1])

            # 6. (Replace)
            sim.population = np.array(new_population)

        # 7. (Result)
        sim.evaluate()
        print(f"{sim.population[0].state}, {sim.population[0].fitness()}")

Simulation.run_simulation(100, 300, 250, 0.005)


