import numpy as np


def noise_mutation(chromosome):
    vector_to_add = np.random.uniform(-1, 1, size=chromosome.shape)
    mutated = vector_to_add + chromosome
    return mutated


def int_mutation(chromosome):

    vector_to_add = np.random.randint(-5, 5, chromosome.shape[0])
    mutated = vector_to_add + chromosome
    return np.where(mutated < 0, 0, mutated)

def mask_mutation(chromosome):
    mask = np.random.rand(*chromosome.shape) < 0.6  
    random_weights = np.random.uniform(-1, 1, chromosome.shape)  
    mutated = np.where(mask, random_weights, chromosome) 
    return mutated