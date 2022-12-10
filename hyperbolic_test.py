
import numpy as np

LOCAL_MAP_PIXEL_WIDTH = 128
hyperbolic_zoom = 2

def map_to_euclidean(pixel_position):
    inverse_dist = 1 - np.hypot(pixel_position[0], pixel_position[1]) / (LOCAL_MAP_PIXEL_WIDTH/2) / hyperbolic_zoom # zoom
    if inverse_dist < 1e-6: # out of range
        inverse_dist = 1e-6
    ret = (2 * pixel_position[0] / inverse_dist,
            2 * pixel_position[1] / inverse_dist)
    return ret

def map_to_hyperbolic(pixel_position):
    dist = np.hypot(pixel_position[..., 0], pixel_position[..., 1])
    c = (LOCAL_MAP_PIXEL_WIDTH/2) * hyperbolic_zoom
    ret = np.array([c * pixel_position[..., 0] / (dist + 2 * c),
                    c * pixel_position[..., 1] / (dist + 2 * c)], dtype=np.float32).T
    return ret

i = np.array([[10, 4],[30, -15],[3, 60]])
euc_coords = np.array([map_to_euclidean(i[0]),map_to_euclidean(i[1]),map_to_euclidean(i[2])])
print(euc_coords)
print(i)

print(map_to_hyperbolic(euc_coords))