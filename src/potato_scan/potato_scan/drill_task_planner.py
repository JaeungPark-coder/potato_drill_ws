"""Pure-python helpers for stage 3: order detected potato eyes into an
efficient visiting sequence (open-path TSP, nearest-neighbor + 2-opt)
and compute drill approach poses from each eye's position + surface
normal (as published by eye_detector on /potato_scan/eye_poses)."""
import numpy as np


def _tour_length(order, points):
    pts = points[order]
    return np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))


def nearest_neighbor_order(points, start_index=0):
    n = len(points)
    visited = [False] * n
    order = [start_index]
    visited[start_index] = True
    for _ in range(n - 1):
        last = points[order[-1]]
        dists = np.linalg.norm(points - last, axis=1)
        dists[visited] = np.inf
        nxt = int(np.argmin(dists))
        order.append(nxt)
        visited[nxt] = True
    return order


def two_opt(order, points, max_passes=50):
    order = list(order)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, len(order) - 1):
            for j in range(i + 1, len(order)):
                candidate = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                if _tour_length(candidate, points) < _tour_length(order, points) - 1e-9:
                    order = candidate
                    improved = True
    return order


def plan_visit_order(positions, start_position=None):
    """positions: (N,3) array of eye positions in base frame.
    Returns a list of indices giving an efficient visiting order,
    starting from whichever eye is closest to `start_position` (e.g. the
    robot's current TCP position) if given."""
    positions = np.asarray(positions, dtype=float)
    n = len(positions)
    if n <= 1:
        return list(range(n))

    if start_position is not None:
        start_index = int(np.argmin(np.linalg.norm(positions - np.asarray(start_position), axis=1)))
    else:
        start_index = 0

    order = nearest_neighbor_order(positions, start_index)
    if n <= 12:  # 2-opt is O(n^2) per pass -- fine for typical eye counts per potato
        order = two_opt(order, positions)
    return order


def approach_pose(position, normal, standoff):
    """Point `standoff` meters back along the outward normal from the
    eye -- the pre-drill approach point that the drill then plunges
    inward from."""
    position = np.asarray(position, dtype=float)
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    return position + normal * standoff
