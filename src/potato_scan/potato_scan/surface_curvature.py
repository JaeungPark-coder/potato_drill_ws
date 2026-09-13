"""Local surface shape from a point cloud: normals, curvature, shape index,
and the clustering that turns the survivors into candidate eyes.

numpy and scipy only -- no Open3D, no ROS. That is deliberate twice over:
this is the part of eye detection worth testing against synthetic surfaces
whose true curvature is known, and Open3D has no wheel for Python 3.13+, so
keeping it out of here is what lets the detector be tested at all on a
machine that cannot install it.

WHY NOT THE OLD CONCAVITY SCORE

The previous detector scored each point by how far its neighbourhood's
centroid sat along its own normal -- an offset, in METRES. Two problems
follow from the units alone:

  * it scales with neighbourhood size, so the threshold has to be retuned
    whenever scan density or `knn` changes, and drifts between setups; and
  * at 0.0015 m it sat in the same range as the sensor's own noise, so the
    threshold was separating signal from noise by magnitude rather than by
    shape.

This module uses two dimensionless quantities instead, which between them
say something the single score could not.

SURFACE VARIATION (kappa), "how curved"

From the eigenvalues of the neighbourhood's covariance, lambda0 <= lambda1
<= lambda2:

    kappa = lambda0 / (lambda0 + lambda1 + lambda2)

Zero on a plane, 1/3 when the neighbourhood is isotropic. Being a ratio it
is dimensionless and bounded, so a threshold on it is a fixed number in a
known range rather than a length that has to be re-derived per setup.

It is NOT scale-invariant, though, and it is worth being precise about that:
with a fixed `knn`, a denser scan means a smaller neighbourhood, which looks
flatter, which lowers kappa. Measured on spheres of one radius sampled at
three densities, the median kappa went 0.0016 -> 0.0006 -> 0.0002 as spacing
went 2.3 -> 1.4 -> 0.9 mm. So kappa still has to be set for the scan density
in use; what it buys over a distance in metres is a bounded, interpretable
range and independence from the sensor's noise floor -- not immunity to
density.

SHAPE INDEX (S), "which way curved"

    S = 1/2 - (1/pi) * arctan((k1 + k2) / (k1 - k2)),   k1 >= k2

from the two principal curvatures. This one IS scale-invariant -- it is a
ratio of curvatures, so it is unchanged by how big the feature is or how
finely it was sampled. It throws away magnitude entirely and keeps only the
KIND of shape, which is what separates an eye from the lump it sits on:

    S ~ 0.00   cup      a pit                <- a potato eye
    S ~ 0.25   rut      a concave trough
    S ~ 0.50   saddle
    S ~ 0.75   ridge    a raised ridge       <- a surface bump
    S ~ 1.00   dome     convex               <- the potato's own body

Curvature alone cannot separate the last two from the first: a sharp ridge
and a sharp pit have similar kappa. Their shape indices are at opposite ends.
So the detector asks for both -- curved enough (kappa) AND cup-shaped (S) --
where it used to ask only "is the neighbourhood centroid offset far enough".

SIGN CONVENTION

Everything here assumes OUTWARD normals, pointing away from the potato's
interior. With those, the potato's own body reads as a dome (S ~ 1) and an
eye as a cup (S ~ 0). Flip the normals and the whole scale inverts, so
estimate_normals takes the potato centre and orients against it rather than
leaving the eigenvector's arbitrary sign in place.
"""
import numpy as np
from scipy.spatial import cKDTree


def neighbor_indices(points, k):
    """Index of each point's k nearest neighbours, excluding itself."""
    points = np.asarray(points, dtype=float)
    k = int(min(k, len(points) - 1))
    if k < 3:
        raise ValueError(f"need at least 3 neighbours, got {k} for {len(points)} points")
    _, idx = cKDTree(points).query(points, k=k + 1)
    return idx[:, 1:]


def neighborhood_eigen(points, idx):
    """Eigenvalues (ascending) and eigenvectors of every neighbourhood's
    covariance, in one batched decomposition."""
    points = np.asarray(points, dtype=float)
    neighbors = points[idx]                              # (N, k, 3)
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    covariance = np.einsum('nki,nkj->nij', centered, centered) / centered.shape[1]
    # eigh, not eig: a covariance is symmetric, and eigh returns real
    # eigenvalues already sorted ascending, which is the order kappa wants.
    return np.linalg.eigh(covariance)


def surface_variation(eigenvalues):
    """kappa = lambda0 / sum(lambda) -- "how curved", dimensionless."""
    total = eigenvalues.sum(axis=1)
    return np.where(total > 0.0, eigenvalues[:, 0] / np.maximum(total, 1e-30), 0.0)


def estimate_normals(points, idx, eigenvectors, outward_from):
    """Unit normals, oriented to point away from `outward_from`.

    The smallest-eigenvalue eigenvector is the neighbourhood's normal but its
    SIGN is arbitrary, and every downstream quantity here depends on it.
    Rather than propagating orientation across the cloud (what Open3D's
    consistent-tangent-plane walk does, and which can flip whole regions on a
    noisy scan), this uses the fact that a potato is star-shaped about its
    own centre: outward is simply the side away from that centre.
    """
    points = np.asarray(points, dtype=float)
    normals = eigenvectors[:, :, 0]                      # column of smallest eigenvalue
    outward = points - np.asarray(outward_from, dtype=float)
    flip = np.einsum('ij,ij->i', normals, outward) < 0.0
    normals = np.where(flip[:, None], -normals, normals)
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


def _tangent_frame(normals):
    """Two unit tangents completing a right-handed frame with each normal."""
    reference = np.where(np.abs(normals[:, 2:3]) < 0.9,
                         np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]))
    t1 = np.cross(reference, normals)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    return t1, np.cross(normals, t1)


def principal_curvatures(points, idx, normals, ridge=1e-12):
    """The two principal curvatures at every point.

    Fits a quadratic height field z = a*x^2 + b*x*y + c*y^2 to each
    neighbourhood in the local frame (normal as z). For such a patch through
    the origin with a horizontal tangent plane, the shape operator is just
    the Hessian [[2a, b], [b, 2c]], whose eigenvalues are the principal
    curvatures -- so no surface-fitting library is needed, only a 3x3 solve
    per point, batched.

    `ridge` regularises neighbourhoods that are degenerate in the tangent
    plane (all neighbours along one line, which happens at a scan's ragged
    edge) instead of raising on a singular matrix.
    """
    points = np.asarray(points, dtype=float)
    t1, t2 = _tangent_frame(normals)

    delta = points[idx] - points[:, None, :]             # (N, k, 3)
    x = np.einsum('nki,ni->nk', delta, t1)
    y = np.einsum('nki,ni->nk', delta, t2)
    z = np.einsum('nki,ni->nk', delta, normals)

    design = np.stack([x * x, x * y, y * y], axis=-1)    # (N, k, 3)
    normal_matrix = np.einsum('nki,nkj->nij', design, design)
    rhs = np.einsum('nki,nk->ni', design, z)

    scale = np.trace(normal_matrix, axis1=1, axis2=2)[:, None, None]
    normal_matrix = normal_matrix + ridge * np.maximum(scale, 1e-30) * np.eye(3)
    # rhs as (N, 3, 1), not (N, 3): numpy 2.x reads a 2-D second operand as a
    # stack of MATRICES, so a (N, 3) right-hand side is taken for one 3xN
    # matrix instead of N vectors.
    a, b, c = np.linalg.solve(normal_matrix, rhs[..., None])[..., 0].T

    # eigenvalues of [[2a, b], [b, 2c]], closed form so it stays vectorised
    mean = a + c
    spread = np.sqrt(np.maximum((a - c) ** 2 + b * b, 0.0))
    return mean + spread, mean - spread                  # k1 >= k2


def shape_index(k1, k2, flat_tolerance=1e-9):
    """Chen & Bhanu's shape index in [0, 1]: 0 is a cup, 1 is a dome.

    Undefined where k1 == k2 exactly, which is both a perfect sphere and a
    perfect plane. Spheres are resolved by their sign (a cup stays 0, a dome
    stays 1); planes, where both curvatures are ~0, are shape-less and are
    reported as 0.5 so they read as neither cup nor dome. In practice the
    kappa test has already removed flat regions before this matters.
    """
    k1 = np.asarray(k1, dtype=float)
    k2 = np.asarray(k2, dtype=float)
    difference = k1 - k2
    total = k1 + k2

    with np.errstate(divide='ignore', invalid='ignore'):
        index = 0.5 - np.arctan(total / difference) / np.pi

    umbilic = difference <= flat_tolerance
    flat = umbilic & (np.abs(total) <= flat_tolerance)
    index = np.where(umbilic, np.where(total > 0.0, 0.0, 1.0), index)
    return np.where(flat, 0.5, index)


def dbscan(points, eps, min_points):
    """DBSCAN labels; -1 is noise.

    Replaces Open3D's cluster_dbscan with the same contract, on a scipy
    KD-tree. `min_points` counts the point itself, matching scikit-learn's
    min_samples (and Open3D's min_points) so the tuned values carry over.
    """
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.empty(0, dtype=int)

    neighbors = cKDTree(points).query_ball_point(points, eps)
    is_core = np.array([len(n) >= min_points for n in neighbors])

    labels = np.full(len(points), -1, dtype=int)
    cluster = 0
    for seed in range(len(points)):
        if labels[seed] != -1 or not is_core[seed]:
            continue
        labels[seed] = cluster
        stack = [seed]
        while stack:
            current = stack.pop()
            for neighbor in neighbors[current]:
                if labels[neighbor] == -1:
                    labels[neighbor] = cluster
                    # only core points extend a cluster; border points join
                    # it but do not carry it further
                    if is_core[neighbor]:
                        stack.append(neighbor)
        cluster += 1
    return labels


def describe_surface(points, potato_center, knn=30):
    """Everything the detector needs about local shape, in one pass over the
    neighbourhood graph: (normals, kappa, shape_index).

    Computed together because all three share the same KD-tree query and the
    same neighbour indices, which dominate the cost.
    """
    idx = neighbor_indices(points, knn)
    eigenvalues, eigenvectors = neighborhood_eigen(points, idx)
    normals = estimate_normals(points, idx, eigenvectors, potato_center)
    kappa = surface_variation(eigenvalues)
    k1, k2 = principal_curvatures(points, idx, normals)
    return normals, kappa, shape_index(k1, k2)


def find_eye_candidates(points, potato_center, knn=30, curvature_min=0.015,
                        shape_index_max=0.35, cluster_eps=0.003,
                        cluster_min_points=8, min_diameter=0.002,
                        max_diameter=0.015):
    """The whole point-cloud half of eye detection, with no ROS in it.

    Selects points that are both curved enough (kappa) and cup-shaped (S),
    clusters the survivors, and keeps clusters whose extent looks like an
    eye. Returns a list of (position, outward_normal, diameter) with the
    strongest-scoring candidates first.

    Two axes rather than one is the substance here: a threshold on curvature
    alone cannot tell a pit from the ridge beside it, because both are
    curved. Adding "and it must be a cup" is what removes the ridges. Point
    level, that still leaves scattered survivors on a lumpy surface -- the
    clustering and the diameter window are what turn those into nothing,
    since scattered points either fail cluster_min_points or merge into a
    blob far wider than max_diameter.

    The two axes do different jobs, measured by ablation on a synthetic
    potato (24000 points on a 35 mm body at ~0.9 mm spacing -- close to what
    a 1 mm voxel grid gives -- with four 3.5 mm-deep dimples):

        both axes      4/4 eyes found, 0 false, worst position error 0.7 mm
        kappa only     4/4 found, 0 false, but error grows to 2.4 mm
        shape only     10 candidates, 1/4 found, 9 false

    So kappa is what rejects, and the shape index is what localises: without
    it the cluster spreads off the cup and the centre drifts. Defaults come
    from that sweep.

    curvature_min has to be re-set for the scan density actually in use (see
    the module docstring on why kappa is not scale-invariant), and 0.015 is
    validated against a synthetic, not against a real potato -- treat it as
    a starting point and check the logged percentiles on a real scan.
    shape_index_max needs no such tuning: 0.35 keeps cups and ruts and
    rejects saddles, ridges and domes at any scale.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < knn + 1:
        return []

    normals, kappa, s_index = describe_surface(points, potato_center, knn=knn)
    selected = (kappa > curvature_min) & (s_index < shape_index_max)
    if not np.any(selected):
        return []

    candidate_points = points[selected]
    labels = dbscan(candidate_points, cluster_eps, cluster_min_points)

    candidates = []
    for label in sorted(set(labels) - {-1}):
        member = labels == label
        cluster = candidate_points[member]
        diameter = float(np.linalg.norm(cluster.max(axis=0) - cluster.min(axis=0)))
        if not (min_diameter <= diameter <= max_diameter):
            continue
        normal = normals[selected][member].mean(axis=0)
        normal /= np.linalg.norm(normal)
        candidates.append({
            'position': cluster.mean(axis=0),
            'normal': normal,
            'diameter': diameter,
            # how cup-like the cluster is on average -- lower is more of a
            # pit, and it is the natural ranking when more candidates come
            # back than a potato plausibly has eyes
            'shape_index': float(s_index[selected][member].mean()),
            'points': int(member.sum()),
        })

    candidates.sort(key=lambda c: c['shape_index'])
    return candidates
