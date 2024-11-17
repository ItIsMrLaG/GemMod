import pytest
from kdtree import *


class TestKDTree:
    cats = [Cat(2, 3, 1), Cat(5, 4, 2), Cat(9, 6, 0), Cat(4, 7, 1), Cat(8, 1, 0), Cat(7, 2, 0)]

    def test_build_kd_tree(self):
        tree = build_kd_tree(self.cats)
        assert tree is not None
        assert tree.cat.point == (7, 2)

    @pytest.mark.parametrize(
        "distance_func, expected1, expected2, expected3",
        [
            (euclidean_distance_squared, Cat(7, 2, 0), Cat(2, 3, 1), Cat(9, 6, 0)),
            (manhattan_distance, Cat(7, 2, 0), Cat(2, 3, 1), Cat(9, 6, 0)),
            (chebyshev_distance, Cat(7, 2, 0), Cat(2, 3, 1), Cat(9, 6, 0))
        ]
    )
    def test_find_nearest_neighbor(self, distance_func, expected1, expected2, expected3):
        tree = build_kd_tree(self.cats)

        target_cat1 = Cat(6, 3, 0)
        nearest1 = find_nearest_neighbor(tree, target_cat1, distance_func)
        assert nearest1 == expected1

        target_cat2 = Cat(1, 1, 0)
        nearest2 = find_nearest_neighbor(tree, target_cat2, distance_func)
        assert nearest2 == expected2

        target_cat3 = Cat(9, 7, 0)
        nearest3 = find_nearest_neighbor(tree, target_cat3, distance_func)
        assert nearest3 == expected3

    @pytest.mark.parametrize("distance_func", [euclidean_distance_squared, manhattan_distance, chebyshev_distance])
    def test_find_nearest_neighbor_empty_tree(self, distance_func):
        tree = build_kd_tree([])
        target_cat = Cat(1, 1, 0)
        nearest = find_nearest_neighbor(tree, target_cat, distance_func)
        assert nearest is None

    @pytest.mark.parametrize("distance_func", [euclidean_distance_squared, manhattan_distance, chebyshev_distance])
    def test_find_nearest_neighbor_single_point(self, distance_func):
        tree = build_kd_tree([Cat(1, 1, 0)])
        target_cat = Cat(1, 1, 0)
        nearest = find_nearest_neighbor(tree, target_cat, distance_func)
        assert nearest is None

    @pytest.mark.parametrize("distance_func", [euclidean_distance_squared, manhattan_distance, chebyshev_distance])
    def test_find_nearest_neighbor_duplicate_points(self, distance_func):
        cats_with_duplicates = [Cat(2, 3, 0), Cat(5, 4, 0), Cat(9, 6, 0), Cat(4, 7, 0), Cat(8, 1, 0), Cat(7, 2, 0), Cat(7, 2, 0)]
        tree = build_kd_tree(cats_with_duplicates)
        target_cat = Cat(6, 3, 0)
        nearest = find_nearest_neighbor(tree, target_cat, distance_func)
        assert nearest == Cat(7, 2, 0)
