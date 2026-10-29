"""VMD read -> write -> read round-trip tests (no external deps)."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vtv.mmd.vmd import parse_vmd
from vtv.mmd.writer import write_vmd, write_identity, _interp_bytes, DEFAULT_CURVE

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VMD_DIR = os.path.join(ROOT, "testdata")
VDMS = [
    os.path.join(VMD_DIR, "R1_妄想天使", "动作.vmd"),
    os.path.join(VMD_DIR, "R2_王牌竞速", "动作＋表情.vmd"),
    os.path.join(VMD_DIR, "R3_花火主播", "火花带表情.vmd"),
]


def _norm(name):
    # VMD names are cp932 and can be byte-truncated by the exporter.  MMD
    # shows an unrepresentable trailing byte as U+FFFD; our writer emits '?',
    # so compare on the normalized form (standard MMD bone names are exact).
    return name.replace("\ufffd", "?")


def _roundtrip(path):
    src = parse_vmd(path)
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "rt.vmd")
        write_vmd(out, src["model_name"], src["bones"], src["morphs"],
                  src["ik_frames"])
        dst = parse_vmd(out)
    return src, dst


class TestVmdRoundtrip(unittest.TestCase):
    def test_interp_layout(self):
        # MMD Tools layout: block b control points at +0,+4,+8,+12
        curves = (((20, 20), (107, 107)),) * 3 + (((49, 0), (107, 107)),)
        raw = _interp_bytes(curves)
        self.assertEqual(len(raw), 64)
        for b in range(4):
            got = (raw[b * 16], raw[b * 16 + 4], raw[b * 16 + 8], raw[b * 16 + 12])
            self.assertEqual(got, curves[b][0] + curves[b][1])

    def _check(self, path):
        if not os.path.exists(path):
            self.skipTest(f"local VMD fixture not redistributed: {path}")
        src, dst = _roundtrip(path)
        self.assertEqual(src["model_name"], dst["model_name"])
        src_bones = {_norm(k): v for k, v in src["bones"].items()}
        dst_bones = {_norm(k): v for k, v in dst["bones"].items()}
        self.assertEqual(set(src_bones), set(dst_bones))
        for name, keys in src_bones.items():
            other = dst_bones[name]
            self.assertEqual(len(keys), len(other), name)
            for a, b in zip(keys, other):
                self.assertEqual(a["frame"], b["frame"])
                self.assertEqual(a["pos"], b["pos"])
                self.assertEqual(a["rot"], b["rot"])
                self.assertEqual(a["curves"], b["curves"])
        src_morphs = {_norm(k): v for k, v in src["morphs"].items()}
        dst_morphs = {_norm(k): v for k, v in dst["morphs"].items()}
        self.assertEqual(src_morphs, dst_morphs)
        self.assertEqual(src["ik_frames"], dst["ik_frames"])
        self.assertEqual(src["max_frame"], dst["max_frame"])

    def test_roundtrip_r1(self):
        self._check(VDMS[0])

    def test_roundtrip_r2(self):
        self._check(VDMS[1])

    def test_roundtrip_r3(self):
        self._check(VDMS[2])

    def test_identity(self):
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "identity.vmd")
            write_identity(out, ["センター", "左足ＩＫ"], ["左足ＩＫ", "左つま先ＩＫ"])
            v = parse_vmd(out)
            self.assertIn("センター", v["bones"])
            self.assertEqual(v["bones"]["センター"][0]["rot"], (0.0, 0.0, 0.0, 1.0))
            self.assertEqual(v["ik_frames"][0]["ik"]["左足ＩＫ"], False)
            self.assertEqual(v["bones"]["センター"][0]["curves"],
                             [(20, 20, 107, 107)] * 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
