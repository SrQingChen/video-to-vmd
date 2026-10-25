"""No-op Wis3D stub.

GVHMR imports wis3d from dataset/debug modules at config-registration time.
Inference never calls it.  Keeping a stub avoids the Wis3D install on Windows;
any actual visualization call becomes a silent no-op (or raises if preferred).
"""


class Wis3D:
    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None
        return _noop
