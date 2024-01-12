import copy
from  network_package.network import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Evolution:
    def __init__(self, crossover_ratio, mutation_ratio, crossover, mutation,
                 selection_func, population_size, input_shape_network, 
                 neurons_num,activations, X, y, loss):
        self.history = {'mean_score': [], 'best_score': [], 'iteration': []}
        self.new_population = None
        self.parents = None
        self.crossover_ratio = crossover_ratio
        self.mutation_ratio = mutation_ratio
        self.crossover = crossover
        self.mutation = mutation
        self.selection_func = selection_func
        self.population_size = population_size

        self.input_shape_network = input_shape_network
        self.neurons_num = neurons_num
        self.activations = activations
        self.population = []
        self.X = X
        self.y = y
        self.loss = loss
        self.initialize_population()
        self.name = f'{crossover.__name__}&{mutation.__name__}&population={population_size}'

    def initialize_population(self):
        for _ in range(self.population_size):
            nn = NN(self.input_shape_network, self.neurons_num, 
                    self.activations, self.X, self.y, self.loss)
            self.population.append(nn)
    
    def do_crossover(self, parentA, parentB):
        child = NN(self.input_shape_network, self.neurons_num, 
                    self.activations, self.X, self.y, self.loss)
        for layer in range(len(parentA.layers)):
            child.layers[layer].weights = self.crossover(parentA.layers[layer].weights, 
                                                 parentB.layers[layer].weights)
        return child

    def do_mutatation(self, network):
        for layer in network.layers:
            mutated_layer_weights = self.mutation(layer.weights)
            layer.weights = mutated_layer_weights
        return network
    
    def fit(self, iterations=20):
        self.name += f'&iterations={iterations}'
        t = 0
        while t < iterations:
            self.new_population = list(self.population)
            self.match_parents()
            for i in range(0, self.population_size, 2):
                if np.random.uniform(size=1) <= self.crossover_ratio:
                    parentA, parentB = self.parents[i], self.parents[i + 1]
                    child = self.do_crossover(parentA, parentB)
                    self.new_population.append(child)

            for i in range(len(self.new_population)):
                if np.random.uniform(size=1) <= self.mutation_ratio:
                    self.new_population[i] = self.do_mutatation(
                        self.new_population[i])
            fitness_scores = self.evaluate()
            self.print_results(fitness_scores, t)
            self.selection(fitness_scores)
            self.print_evaluate_new_generation()
            self.save_to_history(t)
            t += 1
        self.best_individual = self.new_population[np.argmin(fitness_scores)]
    
    def save_to_history(self, iteration):
        scores = []
        for network in self.population:
            fitness = network.calculate_errors()
            scores.append(fitness)
        self.history['mean_score'].append(np.mean(scores))
        self.history['best_score'].append(np.min(scores))
        self.history['iteration'].append(iteration)

    def save_history_to_csv(self):
        df = pd.DataFrame(self.history)
        df.to_csv(f'history\\{self.name}.csv')

    def visualise(self):
        plt.figure(figsize=[10, 6])
        plt.subplot(1, 2, 1)
        plt.plot(self.history['iteration'], self.history['mean_score'])
        plt.title('Mean score of population over time')
        plt.subplot(1, 2, 2)
        plt.plot(self.history['iteration'], self.history['best_score'])
        plt.title('Best score from population over time')
        plt.savefig(f'plots\\{self.name}.png')

    def print_results(self, scores, generation_num):
        print(f'Generation {generation_num} best score is {np.min(scores)}')

    def print_evaluate_new_generation(self):
        scores = []
        for network in self.population:
            fitness = network.calculate_errors()
            scores.append(fitness)
        print(f'Population average score is {np.mean(scores)}\n')

    def match_parents(self):
        self.parents = copy.deepcopy(self.population)
        np.random.shuffle(self.parents)
    
    def evaluate(self):
        fitness_scores = []
        for network in self.new_population:
            fitness = network.calculate_errors()
            fitness_scores.append(fitness)
        return fitness_scores

    def selection(self, scores):
        indices = self.selection_func(scores, self.population_size)
        self.population = np.array(self.new_population)[indices.astype('int')]