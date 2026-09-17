from Individual import Individual
import os
import re
from pathlib import Path
import numpy as np
import random
import json

class Simulation:
    def __init__(self, n : int, m : int):
        """
        Inits the preset values for Simulation, n is the size of the board, m is the size of the population
        """
        self.population = np.array([Individual.random_Individual_legal(n) for _ in range(m)])
        self.n = n
        self.m = m
        self.data = {}

    def evaluate(self):
        fitness = [individual.fitness() for individual in self.population]
        self.population = self.population[np.argsort(fitness)[::-1]]
        return fitness
 
    def select_parents(self):
        parents = []
        for i in range(0, 2 * self.m // 3 - 1, 2):
            parents.append([self.population[i], self.population[i + 1]])
        return np.array(parents)
    def save_data(self, file_path):
        json_string = json.dumps(self.data, indent=4, ensure_ascii=False)
        
        # This keeps each individual chromosome flat, but leaves the outer population array stacked.
        compact_string = re.sub(
            r'\[\s+([^\[\]]*?)\s+\]', 
            lambda m: '[' + re.sub(r'\s+', ' ', m.group(1)).strip() + ']', 
            json_string
        )
        
        # 3. Clean up the trailing commas inside the outer array to keep rows vertically aligned
        compact_string = re.sub(r'\],\s*\n\s*\[', '],\n            [', compact_string)
        
        # 4. Save to disk
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(compact_string)

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

        for i in range(runs):

            # 2. (Evaluate)
            evaluation = sim.evaluate()
            print(f"{sim.population[0].state} fit: {sim.population[0].fitness()}")
            sim.data[f"gen {i}"] = {
                "population": [i.state.tolist() for i in sim.population],
                "evaluation" : evaluation 
            }
            
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
        evaluation = sim.evaluate()
        sim.data["result"] = {
            "population": [i.state.tolist() for i in sim.population],
            "evaluation": evaluation 
        }
        print(f"{sim.population[0].state}, {sim.population[0].fitness()}")
        sim.save_data("data/data.json")

print("Select the size of the board")
n = int(input())
print("Select the size of the population")
m = int(input())
print("Select the number of runs")
runs = int(input())
print("Select the mutation factor (0->1)")
mutation_factor = float(input())
Simulation.run_simulation(n, m, runs, mutation_factor)


