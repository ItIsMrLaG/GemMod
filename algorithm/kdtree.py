import math


class Node:
    def __init__(self, point, axis, left=None, right=None):
        self.point = point
        self.axis = axis
        self.left = left
        self.right = right


def build_kd_tree(points, depth=0):
    if not points:
        return None

    axis = depth % 2
    points.sort(key=lambda x: x[axis])
    median = len(points) // 2

    return Node(
        point=points[median],
        axis=axis,
        left=build_kd_tree(points[:median], depth + 1),
        right=build_kd_tree(points[median + 1:], depth + 1)
    )


def distance(point1, point2):
    return math.sqrt(sum([(x - y) ** 2 for x, y in zip(point1, point2)]))


def find_nearest_neighbor(root, target_point, best=None):
    if root is None:
        return best

    if best is None or distance(root.point, target_point) < distance(best, target_point):
        if root.point != target_point:
            best = root.point

    if target_point[root.axis] < root.point[root.axis]:
        next_branch = root.left
        other_branch = root.right
    else:
        next_branch = root.right
        other_branch = root.left

    best = find_nearest_neighbor(next_branch, target_point, best)

    if (other_branch is not None and
            (best is None or abs(target_point[root.axis] - root.point[root.axis]) < distance(target_point, best))):
        best = find_nearest_neighbor(other_branch, target_point, best)

    return best
