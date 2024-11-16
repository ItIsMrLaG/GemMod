from kdtree import build_kd_tree, find_nearest_neighbor


class TestKDTree:
    points = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]

    def test_build_kd_tree(self):
        tree = build_kd_tree(self.points)
        assert tree is not None
        assert tree.point == (7, 2)

    def test_find_nearest_neighbor(self):
        tree = build_kd_tree(self.points)

        target_point1 = (6, 3)
        nearest1 = find_nearest_neighbor(tree, target_point1)
        assert nearest1 == (7, 2)

        target_point2 = (1, 1)
        nearest2 = find_nearest_neighbor(tree, target_point2)
        assert nearest2 == (2, 3)

        target_point3 = (9, 7)
        nearest3 = find_nearest_neighbor(tree, target_point3)
        assert nearest3 == (9, 6)

    def test_find_nearest_neighbor_empty_tree(self):
        tree = build_kd_tree([])
        target_point = (1, 1)
        nearest = find_nearest_neighbor(tree, target_point)
        assert nearest is None

    def test_find_nearest_neighbor_single_point(self):
        tree = build_kd_tree([(1, 1)])
        target_point = (1, 1)
        nearest = find_nearest_neighbor(tree, target_point)
        assert nearest is None

    def test_find_nearest_neighbor_duplicate_points(self):
        points_with_duplicates = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2), (7, 2)]
        tree = build_kd_tree(points_with_duplicates)
        target_point = (6, 3)
        nearest = find_nearest_neighbor(tree, target_point)
        assert nearest == (7, 2)
