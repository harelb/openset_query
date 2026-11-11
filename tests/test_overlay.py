import numpy as np
from PIL import Image

from openset_query.overlay import annotate_rgb, render_annotated_data_uri


def _write_mask(path, shape, box):
    m = np.zeros(shape[:2], dtype=np.uint8)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = 255
    Image.fromarray(m).save(path)


def test_annotate_rgb_tints_only_masked_pixels(tmp_path):
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    mask_path = tmp_path / "m.png"
    _write_mask(mask_path, rgb.shape, (5, 5, 10, 10))
    out = annotate_rgb(rgb, mask_path=str(mask_path), color=(255, 0, 0), alpha=1.0)
    assert tuple(out[7, 7]) == (255, 0, 0)  # inside mask -> tinted
    assert tuple(out[0, 0]) == (0, 0, 0)  # outside mask -> untouched
    assert out.dtype == np.uint8


def test_annotate_rgb_draws_clipped_bbox(tmp_path):
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    out = annotate_rgb(rgb, bbox=(2, 2, 15, 15), color=(0, 255, 0), thickness=1)
    assert tuple(out[2, 8]) == (0, 255, 0)  # top edge drawn
    assert tuple(out[10, 10]) == (0, 0, 0)  # interior untouched


def test_annotate_rgb_noop_returns_copy(tmp_path):
    rgb = np.full((4, 4, 3), 9, dtype=np.uint8)
    out = annotate_rgb(rgb)
    assert np.array_equal(out, rgb)
    out[0, 0] = 0
    assert rgb[0, 0, 0] == 9  # original not mutated


def test_annotate_rgb_resizes_mismatched_mask(tmp_path):
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    mask_path = tmp_path / "m.png"
    _write_mask(mask_path, (10, 10), (0, 0, 10, 10))  # half-size, all-on
    out = annotate_rgb(rgb, mask_path=str(mask_path), color=(255, 0, 0), alpha=1.0)
    assert tuple(out[15, 15]) == (255, 0, 0)  # scaled up to cover full image


def test_render_annotated_data_uri_uses_the_observation_mask_and_bbox(tmp_path):
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    Image.fromarray(rgb).save(tmp_path / "frame_123_rgb.jpg")
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 5:10] = True
    res = render_annotated_data_uri(
        str(tmp_path / "frame_123_rgb.jpg"), mask=mask, bbox=(5, 5, 10, 10)
    )
    assert res["annotated"] is True
    assert res["uri"].startswith("data:image/jpeg;base64,")


def test_annotate_rgb_accepts_a_boolean_mask_array():
    rgb = np.zeros((20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 5:10] = True
    out = annotate_rgb(rgb, mask=mask, color=(255, 0, 0), alpha=1.0)
    assert tuple(out[7, 7]) == (255, 0, 0)
    assert tuple(out[15, 15]) == (0, 0, 0)


def test_render_annotated_data_uri_plain_without_mask_or_bbox(tmp_path):
    Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8)).save(
        tmp_path / "frame_9_rgb.jpg"
    )
    res = render_annotated_data_uri(str(tmp_path / "frame_9_rgb.jpg"))
    assert res["annotated"] is False
    assert res["uri"].startswith("data:image/jpeg;base64,")
