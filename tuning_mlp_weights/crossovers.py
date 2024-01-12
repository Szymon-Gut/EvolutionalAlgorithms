import numpy as np


def one_point_crossover(parentA, parentB):
    n = parentA.shape[0]
    split_at = n // 2
    child = np.zeros(parentA.shape)

    child[:split_at] = parentA[:split_at]
    child[split_at:] = parentB[split_at:]

    return child


def mean_crossover(parentA, parentB):
    return (parentA + parentB) // 2

def one_point_multi_dim(layerA, layerB):
    mask = np.random.randint(2, size=layerA.shape).astype(bool)
    child_layer = np.where(mask, layerA, layerB)
    return child_layer

def random_layer(layerA, layerB):
    prob = np.random.randint(10)
    if prob % 2 == 0:
        return layerA
    else:
        return layerB