"""Build a real Hydra image store (via the ``hydra_python.image_store`` writer) for tests.

Replaces the old fake ``image_folder`` trees: frames, depth, observations (kept /
unkept, with masks and boxes) and merges all go through the package writer, so the
tests exercise the same reader code the consumers use.
"""

from __future__ import annotations

import pathlib

import numpy as np
from PIL import Image

from hydra_python.image_store import FrameKind, ImageStore, make_node_id

W, H = 64, 48


def node_id(symbol: str) -> int:
    """``O605`` -> spark_dsg node id."""
    return make_node_id(symbol[0], int("".join(c for c in symbol if c.isdigit())))


class StoreBuilder:
    def __init__(self, root, robot_id: int = 0, size=(W, H)):
        self.root = pathlib.Path(root)
        self.size = size
        self.w = ImageStore.open_writer(self.root, robot_id=robot_id)
        self.sensor = self.w.add_sensor(
            "cam",
            size[0],
            size[1],
            fx=50.0,
            fy=50.0,
            cx=size[0] / 2,
            cy=size[1] / 2,
            depth_scale=0.001,
            body_T_sensor=np.eye(4),
        )
        self._tracks = 100

    def frame(
        self,
        ts: int,
        color=(0, 255, 0),
        depth: bool = True,
        kind=FrameKind.KEYFRAME,
        anchor: int | None = None,
        pose=None,
        depth_value: int = 2000,
        rgb: bool = True,
        anchor_T_body=None,
    ) -> int:
        """A frame; an anchored keyframe is its anchor's own frame unless
        ``anchor_T_body`` is not the identity (then an "other" keyframe, what ADT4
        calls a sub-keyframe)."""
        base = self.root / "frames" / f"r{self.w.robot_id}"
        base.mkdir(parents=True, exist_ok=True)
        rgb_p = dep_p = None
        if rgb:
            rgb_p = base / f"{ts}_rgb.jpg"
            Image.new("RGB", self.size, color).save(rgb_p, format="JPEG", quality=95)
        if depth:
            dep_p = base / f"{ts}_depth.png"
            Image.fromarray(
                np.full(self.size[::-1], depth_value, dtype=np.uint16)
            ).save(dep_p)
        return self.w.add_frame_ref(
            self.sensor,
            ts,
            kind=kind,
            world_T_body_odom=pose,
            rgb=rgb_p,
            depth=dep_p,
            anchor_node_id=anchor,
            anchor_T_body=anchor_T_body,
        )

    def observation(
        self,
        symbol: str,
        frame_id: int,
        bbox=None,
        mask_box=None,
        score: float = 0.5,
        kept: bool = True,
        track: int | None = None,
        ts: int | None = None,
    ) -> int:
        """Observation of object ``symbol`` (a Khronos-style track bound to the node)."""
        node = node_id(symbol)
        if track is None:
            self._tracks += 1
            track = self._tracks
        self.w.bind_track(track, node)
        mask = None
        if mask_box is not None:
            x0, y0, x1, y1 = mask_box
            mask = np.zeros(self.size[::-1], dtype=bool)
            mask[y0:y1, x0:x1] = True
        return self.w.add_observation(
            track,
            frame_id,
            timestamp_ns=ts,
            bbox2d=bbox,
            mask=mask,
            score=score,
            kept=kept,
        )

    def close(self) -> pathlib.Path:
        self.w.close()
        return self.root


def off_anchor(dx: float = 0.5) -> np.ndarray:
    """A non-identity ``anchor_T_body``: a keyframe taken away from its anchor's stamp."""
    T = np.eye(4)
    T[0, 3] = dx
    return T


def build_store(root, **kw) -> StoreBuilder:
    return StoreBuilder(root, **kw)
