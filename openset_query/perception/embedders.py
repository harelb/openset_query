"""Image embedder protocols and concrete implementations.

Supported backends:
  - StubEmbedder      — deterministic hash-based embedder, no dependencies.
  - OpenClipEmbedder  — wraps open_clip (requires: pip install open_clip_torch torch).
  - SigLip2Embedder   — wraps HuggingFace transformers (requires: pip install transformers torch).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, Sequence, Union

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

ImageRef = Union[str, Path, "Image.Image"]

# ---------------------------------------------------------------------------
# Optional heavy imports — guarded so the module always loads successfully.
# ---------------------------------------------------------------------------

try:
    import torch
    import open_clip

    _OPEN_CLIP_AVAILABLE = True
    _OPEN_CLIP_IMPORT_ERROR = None
except ImportError as _e:
    _OPEN_CLIP_AVAILABLE = False
    _err = ImportError(
        "OpenClipEmbedder requires open_clip and torch. "
        "Install with: pip install open_clip_torch torch"
    )
    _err.__cause__ = _e
    _OPEN_CLIP_IMPORT_ERROR = _err

try:
    import torch as _torch_transformers
    from transformers import AutoModel, AutoProcessor

    _TRANSFORMERS_AVAILABLE = True
    _TRANSFORMERS_IMPORT_ERROR = None
except ImportError as _e:
    _TRANSFORMERS_AVAILABLE = False
    _err = ImportError(
        "SigLip2Embedder requires transformers and torch. "
        "Install with: pip install transformers torch"
    )
    _err.__cause__ = _e
    _TRANSFORMERS_IMPORT_ERROR = _err


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class ImageEmbedder(Protocol):
    """Embedding model that produces aligned image and text embeddings."""

    name: str
    dim: int

    def embed_images(self, images: Sequence[ImageRef]) -> np.ndarray:
        """Return an (N, dim) L2-normalized float32 numpy array."""
        ...

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (N, dim) L2-normalized float32 numpy array."""
        ...


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    """Row-wise L2-normalize a 2-D float32 array."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1.0, norms)
    return (arr / norms).astype(np.float32)


def _hash_to_vector(data: bytes, dim: int) -> np.ndarray:
    """Convert arbitrary bytes to a deterministic *dim*-length float32 vector.

    Strategy:
      1. Compute SHA-256 (32 bytes).
      2. Tile to at least *dim* bytes.
      3. Interpret as uint8, cast to float32, subtract 127.5 (centre around 0).
      4. Return the first *dim* elements (before normalisation).
    """
    digest = hashlib.sha256(data).digest()  # 32 bytes
    # Tile enough times to cover dim bytes
    repeats = (dim + len(digest) - 1) // len(digest)
    tiled = (digest * repeats)[:dim]
    raw = np.frombuffer(tiled, dtype=np.uint8).astype(np.float32) - 127.5
    return raw


def _load_image(ref: ImageRef) -> Image.Image:
    """Resolve an ImageRef to a PIL.Image.Image."""
    if isinstance(ref, Image.Image):
        return ref
    return Image.open(ref)


def _image_to_canonical_bytes(img: Image.Image) -> bytes:
    """Convert a PIL image to a deterministic byte representation."""
    return img.convert("RGB").resize((32, 32)).tobytes()


def _unpack_model_output(out, torch_mod):
    """Extract a plain Tensor from a ModelOutput or tuple returned by HF models."""
    if isinstance(out, torch_mod.Tensor):
        return out
    pooler = getattr(out, "pooler_output", None)
    if pooler is not None:
        return pooler
    last = getattr(out, "last_hidden_state", None)
    if last is not None:
        # mean-pool sequence dim
        return last.mean(dim=1)
    # Fallback: first element of tuple/list
    return out[0]


# ---------------------------------------------------------------------------
# StubEmbedder
# ---------------------------------------------------------------------------


class StubEmbedder:
    """Deterministic hash-based embedder. No ML dependencies required.

    Parameters
    ----------
    dim:
        Embedding dimensionality.
    name:
        Identifier string.
    """

    def __init__(self, dim: int = 64, name: str = "stub") -> None:
        self.dim = dim
        self.name = name

    def embed_images(self, images: Sequence[ImageRef]) -> np.ndarray:
        """Return an (N, dim) L2-normalized float32 array."""
        vectors = []
        for ref in images:
            img = _load_image(ref)
            raw = _hash_to_vector(_image_to_canonical_bytes(img), self.dim)
            vectors.append(raw)
        return _l2_normalize(np.stack(vectors, axis=0))

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (N, dim) L2-normalized float32 array."""
        vectors = []
        for text in texts:
            raw = _hash_to_vector(text.encode("utf-8"), self.dim)
            vectors.append(raw)
        return _l2_normalize(np.stack(vectors, axis=0))


# ---------------------------------------------------------------------------
# OpenClipEmbedder
# ---------------------------------------------------------------------------


class OpenClipEmbedder:
    """Image/text embedder backed by OpenCLIP.

    Parameters
    ----------
    model_name:
        OpenCLIP model architecture, e.g. ``"ViT-H-14"``.
    pretrained:
        Pre-trained weights tag, e.g. ``"laion2b_s32b_b79k"``.
    device:
        ``"auto"`` selects CUDA when available, otherwise CPU.

    Raises
    ------
    ImportError
        If ``open_clip`` or ``torch`` are not installed.
    """

    def __init__(
        self,
        model_name: str = "ViT-H-14",
        pretrained: str = "laion2b_s32b_b79k",
        device: str = "auto",
    ) -> None:
        if not _OPEN_CLIP_AVAILABLE:
            raise _OPEN_CLIP_IMPORT_ERROR  # type: ignore[misc]

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = device

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        model.eval()
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(model_name)

        self.name = f"openclip:{model_name}:{pretrained}"
        self.dim = self._resolve_dim(model)

    @staticmethod
    def _resolve_dim(model) -> int:  # type: ignore[no-untyped-def]
        """Return the output embedding dimension of the model."""
        try:
            return int(model.visual.output_dim)
        except AttributeError:
            pass
        # Fallback: probe with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out = model.encode_image(dummy)
        return int(out.shape[-1])

    def embed_images(self, images: Sequence[ImageRef]) -> np.ndarray:
        """Return an (N, dim) L2-normalized float32 array."""
        tensors = []
        for ref in images:
            img = _load_image(ref)
            tensors.append(self._preprocess(img))
        batch = torch.stack(tensors).to(self._device)
        with torch.no_grad():
            features = self._model.encode_image(batch).float().cpu().numpy()
        return _l2_normalize(features)

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (N, dim) L2-normalized float32 array."""
        tokens = self._tokenizer(list(texts)).to(self._device)
        with torch.no_grad():
            features = self._model.encode_text(tokens).float().cpu().numpy()
        return _l2_normalize(features)


# ---------------------------------------------------------------------------
# SigLip2Embedder
# ---------------------------------------------------------------------------


class SigLip2Embedder:
    """Image/text embedder backed by SigLIP-2 via HuggingFace Transformers.

    Parameters
    ----------
    model_id:
        HuggingFace model hub ID.
    device:
        ``"auto"`` selects CUDA when available, otherwise CPU.

    Raises
    ------
    ImportError
        If ``transformers`` or ``torch`` are not installed.
    """

    def __init__(
        self,
        model_id: str = "google/siglip2-so400m-patch16-naflex",
        device: str = "auto",
    ) -> None:
        if not _TRANSFORMERS_AVAILABLE:
            raise _TRANSFORMERS_IMPORT_ERROR  # type: ignore[misc]

        if device == "auto":
            device = "cuda" if _torch_transformers.cuda.is_available() else "cpu"
        self._device = device

        self._model = AutoModel.from_pretrained(model_id).to(device).eval()
        self._processor = AutoProcessor.from_pretrained(model_id)

        self.name = f"siglip2:{model_id}"
        self.dim = self._resolve_dim(self._model, self._processor)

    @staticmethod
    def _resolve_dim(model, processor) -> int:  # type: ignore[no-untyped-def]
        """Infer the output embedding dimension from the model config."""
        cfg = getattr(model, "config", None)
        if cfg is not None:
            # Try vision projection dim first, then text projection dim
            for attr in (
                "vision_config.projection_dim",
                "text_config.projection_dim",
                "projection_dim",
            ):
                try:
                    val = cfg
                    for part in attr.split("."):
                        val = getattr(val, part)
                    return int(val)
                except AttributeError:
                    continue
        # Fallback: probe with a dummy PIL image through the processor so that
        # variable-resolution models (e.g. naflex) receive correctly shaped inputs.
        import torch as _t
        from PIL import Image as _PILImage  # noqa: PLC0415
        dummy_img = _PILImage.new("RGB", (224, 224))
        inputs = processor(images=[dummy_img], return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, _t.Tensor)}
        with _t.no_grad():
            out = model.get_image_features(**inputs)
        tensor_out = _unpack_model_output(out, _t)
        return int(tensor_out.shape[-1])

    def embed_images(self, images: Sequence[ImageRef]) -> np.ndarray:
        """Return an (N, dim) L2-normalized float32 array."""
        pil_images = [_load_image(ref) for ref in images]
        inputs = self._processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items() if isinstance(v, _torch_transformers.Tensor)}
        with _torch_transformers.no_grad():
            out = self._model.get_image_features(**inputs)
        features = _unpack_model_output(out, _torch_transformers).float().cpu().numpy()
        return _l2_normalize(features)

    def embed_text(self, texts: Sequence[str]) -> np.ndarray:
        """Return an (N, dim) L2-normalized float32 array."""
        inputs = self._processor(
            text=list(texts), return_tensors="pt", padding=True, truncation=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items() if isinstance(v, _torch_transformers.Tensor)}
        with _torch_transformers.no_grad():
            out = self._model.get_text_features(**inputs)
        features = _unpack_model_output(out, _torch_transformers).float().cpu().numpy()
        return _l2_normalize(features)
