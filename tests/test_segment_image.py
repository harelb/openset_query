"""``POST /segment``: SAM3 on one uploaded image (stub segmenter, no GPU).

A real-model run is marked ``gpu`` (OPENSET_QUERY_MODEL_TESTS=1)."""

import base64
import io
import json
import os
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

from openset_query import query_server as qs
from openset_query import segment_image as si
from openset_query.sam3_segmenter import Detection


def _jpeg(w=64, h=48) -> bytes:
    img = np.zeros((h, w, 3), np.uint8)
    img[10:30, 20:40] = (200, 30, 30)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG")
    return buf.getvalue()


class StubSegmenter:
    def __init__(self):
        self.calls = []

    def segment(self, image, prompts):
        self.calls.append((image.shape, list(prompts)))
        m = np.zeros(image.shape[:2], bool)
        m[10:30, 20:40] = True
        weak = np.zeros_like(m)
        weak[0:2, 0:2] = True
        return [
            Detection(
                label=prompts[0], mask=weak, box=np.array([0, 0, 2, 2]), score=0.2
            ),
            Detection(
                label=prompts[0], mask=m, box=np.array([20, 10, 40, 30]), score=0.9
            ),
        ]


def test_segment_sorts_filters_and_describes_detections():
    seg = StubSegmenter()
    out = si.segment(
        seg,
        {
            "image_b64": base64.b64encode(_jpeg()).decode(),
            "prompt": " red box ",
            "conf": 0.5,
        },
    )
    assert seg.calls == [((48, 64, 3), ["red box"])]
    assert out["ok"] and (out["width"], out["height"]) == (64, 48)
    (d,) = out["detections"]
    assert d["score"] == 0.9 and d["box_xyxy"] == [20.0, 10.0, 40.0, 30.0]
    assert d["center_px"] == [30.0, 20.0] and d["area_px"] == 400
    assert d["mask_rle"]["size"] == [48, 64]
    ov = Image.open(io.BytesIO(base64.b64decode(out["overlay_jpeg_b64"])))
    assert ov.size == (64, 48)


def test_segment_accepts_a_data_uri_and_skips_the_overlay():
    uri = "data:image/jpeg;base64," + base64.b64encode(_jpeg()).decode()
    out = si.segment(
        StubSegmenter(),
        {"image_b64": uri, "prompt": "box", "overlay": False, "max_detections": 1},
    )
    assert len(out["detections"]) == 1 and out["detections"][0]["score"] == 0.9
    assert out["overlay_jpeg_b64"] is None


def test_mask_rle_without_pycocotools_round_trips(monkeypatch):
    import builtins

    real = builtins.__import__

    def no_coco(name, *a, **kw):
        if name.startswith("pycocotools"):
            raise ImportError(name)
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_coco)
    m = np.zeros((3, 4), bool)
    m[1, 1:3] = True
    rle = si.mask_rle(m)
    flat = np.concatenate(
        [np.full(n, i % 2, np.uint8) for i, n in enumerate(rle["counts"])]
    )
    assert np.array_equal(flat.reshape((3, 4), order="F").astype(bool), m)


@pytest.mark.parametrize(
    "req, msg",
    [
        ({"prompt": "x"}, "image_b64 is required"),
        ({"image_b64": "@@@", "prompt": "x"}, "cannot decode"),
        (
            {"image_b64": base64.b64encode(b"not an image").decode(), "prompt": "x"},
            "cannot decode",
        ),
        ({"image_b64": "aGk=", "prompt": "  "}, "empty prompt"),
        ({"image_b64": "aGk=", "prompt": "x", "conf": 2}, "conf must be"),
    ],
)
def test_bad_requests(req, msg):
    with pytest.raises(si.SegmentRequestError, match=msg):
        si.parse_request(req)


class _Engine:
    def __init__(self):
        self.seg = StubSegmenter()
        self.lock = threading.Lock()

    def segment_upload(self, req):
        with self.lock:
            return si.segment(self.seg, req)


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), qs.make_handler(_Engine()))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _post(url, body):
    req = urllib.request.Request(url + "/segment", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")  # fmt: skip
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_http_route(server):
    code, out = _post(
        server, {"image_b64": base64.b64encode(_jpeg()).decode(), "prompt": "box"}
    )
    assert code == 200 and len(out["detections"]) == 2
    code, out = _post(server, {"image_b64": "", "prompt": "box"})
    assert code == 400 and out["ok"] is False


@pytest.mark.gpu
def test_real_sam3_on_an_image():
    from openset_query.config import Sam3FrontendConfig
    from openset_query.sam3_segmenter import Sam3Segmenter

    path = os.environ.get("SEGMENT_TEST_IMAGE")
    if not path:
        pytest.skip("set SEGMENT_TEST_IMAGE to an outdoor JPEG")
    seg = Sam3Segmenter(Sam3FrontendConfig())
    with open(path, "rb") as f:
        out = si.segment(seg, {"image_b64": base64.b64encode(f.read()).decode(),
                               "prompt": os.environ.get("SEGMENT_TEST_PROMPT", "tree")})  # fmt: skip
    assert out["ok"] and out["detections"], out
