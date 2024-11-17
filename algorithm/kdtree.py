import math

from cat import Cat


class Node:
    def __init__(self, cat: Cat, axis, left=None, right=None):
        self.cat = cat
        self.axis = axis
        self.left = left
        self.right = right


def build_kd_tree(cats, depth=0):
    if not cats:
        return None

    axis = depth % 2
    cats.sort(key=lambda cat: cat.point[axis])
    median = len(cats) // 2

    return Node(
        cat=cats[median],
        axis=axis,
        left=build_kd_tree(cats[:median], depth + 1),
        right=build_kd_tree(cats[median + 1:], depth + 1)
    )


def euclidean_distance_squared(point1, point2):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))


def manhattan_distance(point1, point2):
    return sum(abs(x - y) for x, y in zip(point1, point2))


def chebyshev_distance(point1, point2):
    return max(abs(x - y) for x, y in zip(point1, point2))


def find_nearest_neighbor(root: Node, target_cat: Cat, distance_func=euclidean_distance_squared, best=None) -> Cat | None:
    if root is None:
        return best

    target_point = target_cat.point
    root_point = root.cat.point

    if best is None or distance_func(root_point, target_point) < distance_func(best.point, target_point):
        if root_point != target_point:
            best = root.cat

    if target_point[root.axis] < root_point[root.axis]:
        next_branch = root.left
        other_branch = root.right
    else:
        next_branch = root.right
        other_branch = root.left

    best = find_nearest_neighbor(next_branch, target_cat, distance_func, best)

    if (other_branch is not None and
            (best is None or (target_point[root.axis] - root_point[root.axis]) < distance_func(target_point, best.point))):
        best = find_nearest_neighbor(other_branch, target_cat, distance_func, best)

    return best

