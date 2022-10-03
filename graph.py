import numpy as np
from matplotlib import pyplot as plt
import skfmm
from skimage import morphology, filters
from scipy import signal

import spfa

def generate_obstacles(shape, num, obstacle_shape):
    highY = shape[0] - obstacle_shape[0]
    highX = shape[1] - obstacle_shape[1]
    randY = np.random.uniform(low=0, high=highY, size=(num,))
    randX = np.random.uniform(low=0, high=highX, size=(num,))
    height = obstacle_shape[0]
    width = obstacle_shape[0]

    map = np.zeros(shape, dtype=bool)
    for y, x in zip(randY, randX):
        map[int(y):int(y+height), int(x):int(x+width)] = True
    return map

if __name__ == "__main__":
    obstacles = generate_obstacles((500, 500), 20, (30, 30))
    distance_field = skfmm.distance(~obstacles)
    peaks = morphology.thin(distance_field) == 1

    kernel = np.ones((5, 5)) # Neighbor-counting
    neighbors = signal.convolve(peaks, kernel, mode='same') > 6

    fig = plt.figure(figsize=(20, 9))
    axes = fig.subplots(1, 5) 
    axes[0].imshow(obstacles) 
    axes[1].imshow(distance_field)
    axes[2].imshow(peaks)
    axes[3].imshow(neighbors)

    kernel = np.ones((13, 13))
    axes[4].imshow(morphology.binary_dilation(neighbors, selem=kernel) + peaks + obstacles)
    plt.pause(1)