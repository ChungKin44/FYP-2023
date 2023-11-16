import numpy as np
import math

class ConvexHull:
    def __init__(self):
        self.points = []

    def add_point(self, point):
        self.points.append(point.flatten())

    def orientation(self, p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < 1e-9:
            return 0  # collinear
        elif val > 0:
            return 1  # clockwise
        else:
            return 2  # counterclockwise

    def polar_angle(self, p, q):
        dx = q[0] - p[0]
        dy = q[1] - p[1]
        return math.atan2(dy, dx)

    def convex_hull(self):
        n = len(self.points)
        if n < 3:
            return []

        hull = []

        # Find the leftmost point
        sorted_indices = np.lexsort((np.array(self.points)[:, 1], np.array(self.points)[:, 0]))
        leftmost_index = sorted_indices[0]
        leftmost = self.points[leftmost_index]

        # Sort points by polar angle
        sorted_points = np.array(self.points)[sorted_indices]

        p = 0  # Index of the current point
        q = None

        while True:
            hull.append(sorted_points[p])
            q = (p + 1) % n

            for r in range(n):
                if self.orientation(sorted_points[p], sorted_points[q], sorted_points[r]) == 2:
                    q = r

            p = q

            if p == 0:
                break

        return hull

    def compute_convex_hull(self, leaf_set):
        for point in leaf_set:
            self.add_point(point)
        return self.convex_hull()

