class Cat:
    point: (int, int)
    status: int

    def __init__(self, x, y, state):
        self.point = (x, y)
        self.status = state

    def __eq__(self, other):
        if isinstance(other, Cat):
            return self.point == other.point and self.status == other.status
        return False
