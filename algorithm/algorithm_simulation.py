import random
from kdtree import find_nearest_neighbor, euclidean_distance_squared, manhattan_distance, chebyshev_distance
from cat import Cat


# def cat_check(cat):
#     pass


def find_neighbor_for_cat(cat: Cat, tree, distance_func, r0, R0):
    neighbor = find_nearest_neighbor(tree, cat, distance_func)

    # # TODO: добавить провекру, что сосед -- это не угол дома
    # if not cat_check(neighbor):
    #     pass

    distance = distance_func(cat.point, neighbor)
    if distance <= r0:
        cat.status = 2
        neighbor.status = 2
    elif distance <= R0 and random.choices([0, 1], weights=[1 - 1 / (distance ** 2), 1 / (distance ** 2)], k=1)[0]:
        cat.status = max(cat.status, 1)
        neighbor.status = max(neighbor.status, 1)
    else:
        cat.status = 0


def find_neighbors(tree, cats: list[Cat], distance_type, r0, R0):
    if distance_type == 'euclidean':
        distance_func = euclidean_distance_squared
    elif distance_type == 'manhattan':
        distance_func = manhattan_distance
    elif distance_type == 'chebyshev':
        distance_func = chebyshev_distance
    else:
        raise ValueError("Unsupported distance type")

    for cat in cats:
        find_neighbor_for_cat(cat, tree, distance_func, r0, R0)

