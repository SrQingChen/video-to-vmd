"""KNN stub.

GVHMR imports this module but has no inference-time caller of knn_points.
Raise loudly instead of silently returning wrong neighbours.
"""


def knn_points(*args, **kwargs):
    raise NotImplementedError(
        "pytorch3d.ops.knn_points is not available in the vtv shim; "
        "GVHMR inference does not need it.")
