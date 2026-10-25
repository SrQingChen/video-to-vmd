"""Import-only PyTorch3D renderer stub.

GVHMR's config registration imports training dataset modules, which import the
renderer at module load.  VTV inference never instantiates a renderer, so these
placeholders only need to exist; calling them raises.
"""


class _Unavailable:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "pytorch3d.renderer is not available in the vtv shim; "
            "this path is visualization-only.")

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("pytorch3d.renderer stub")


PerspectiveCameras = _Unavailable
TexturesVertex = _Unavailable
PointLights = _Unavailable
Materials = _Unavailable
RasterizationSettings = _Unavailable
MeshRenderer = _Unavailable
MeshRasterizer = _Unavailable
SoftPhongShader = _Unavailable
