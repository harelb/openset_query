"""Tests for openset_query.perception.embedders."""

import numpy as np
import pytest
from PIL import Image

from openset_query.perception.embedders import (
    StubEmbedder,
    OpenClipEmbedder,
    SigLip2Embedder,
)


# ---------------------------------------------------------------------------
# StubEmbedder — no heavy deps, always run
# ---------------------------------------------------------------------------


def test_stub_embed_text_deterministic():
    e = StubEmbedder(dim=64)
    v1 = e.embed_text(["hello"])
    v2 = e.embed_text(["hello"])
    np.testing.assert_array_equal(v1, v2)


def test_stub_embed_text_distinct():
    e = StubEmbedder(dim=64)
    v = e.embed_text(["hello", "world"])
    assert v.shape == (2, 64)
    # Different inputs → different vectors
    assert not np.array_equal(v[0], v[1])


def test_stub_embed_normalized():
    e = StubEmbedder(dim=64)
    v = e.embed_text(["hello", "world"])
    norms = np.linalg.norm(v, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_stub_embed_image(tmp_path):
    p = tmp_path / "img.png"
    Image.new("RGB", (64, 64), color=(255, 0, 0)).save(p)
    e = StubEmbedder(dim=64)
    v = e.embed_images([p])
    assert v.shape == (1, 64)
    assert abs(np.linalg.norm(v[0]) - 1.0) < 1e-5


def test_stub_embed_image_accepts_pil():
    img = Image.new("RGB", (64, 64), color=(0, 255, 0))
    e = StubEmbedder(dim=64)
    v = e.embed_images([img])
    assert v.shape == (1, 64)


# ---------------------------------------------------------------------------
# OpenClipEmbedder — requires open_clip + torch + model weights
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_openclip_smoke():
    e = OpenClipEmbedder(
        model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device="cpu"
    )
    v_text = e.embed_text(["a photo of a chair"])
    v_img = e.embed_images([Image.new("RGB", (224, 224), color=(128, 128, 128))])
    assert v_text.shape[1] == e.dim
    assert v_img.shape[1] == e.dim


# ---------------------------------------------------------------------------
# SigLip2Embedder — requires transformers + torch + model weights
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_siglip2_smoke():
    e = SigLip2Embedder(device="cpu")  # any small siglip2 if default is too big
    v_text = e.embed_text(["a photo of a chair"])
    assert v_text.shape[1] == e.dim
