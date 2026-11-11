"""CLIP / SigLIP2 frame pre-filter.

Ranks keyframes by text-image cosine similarity to the query prompt(s) using a
cheap vision-language model, so the heavy SAM3 model only runs on the top-K most
relevant frames. Image embeddings are cached to disk (keyframes don't change), so
the per-query cost is just text encoding + a matmul after the first pass.

SigLIP cosine values are small in absolute terms (sigmoid-trained), so selection
is by top-K ranking by default; an optional threshold can also be applied.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger("openset_query.embedding_filter")

# Both are present in the HF cache on this machine.
MODEL_ALIASES = {
    "siglip2": "google/siglip2-so400m-patch16-naflex",
    "clip": "openai/clip-vit-base-patch32",
}


class EmbeddingFilter:
    """Text-image retrieval pre-filter backed by a transformers VLM (SigLIP2/CLIP)."""

    def __init__(
        self,
        model: str = "siglip2",
        device: str = "cuda",
        cache_path: Optional[str] = None,
        batch_size: int = 32,
    ):
        self.model_id = MODEL_ALIASES.get(model, model)
        self.device = device
        self.batch_size = batch_size
        self.cache_path = cache_path
        self._model = None
        self._proc = None
        self._torch = None
        self._cache = {}  # key (frame_id, else rgb_path) -> np.ndarray (L2-normalized)
        if cache_path and os.path.exists(cache_path):
            self._load_cache()

    def _ensure(self):
        if self._model is not None:
            return
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # models are cached
        import torch
        from transformers import AutoModel, AutoProcessor

        self._torch = torch
        self._model = AutoModel.from_pretrained(self.model_id).to(self.device).eval()
        self._proc = AutoProcessor.from_pretrained(self.model_id)
        logger.info("Loaded embedding model %s", self.model_id)

    @staticmethod
    def _feat(out):
        import torch

        if isinstance(out, torch.Tensor):
            return out
        for a in ("image_embeds", "text_embeds", "pooler_output"):
            v = getattr(out, a, None)
            if v is not None:
                return v
        return out[0]

    def _load_cache(self):
        try:
            d = np.load(self.cache_path, allow_pickle=True)
            # np.load on an .npz is lazy: every d["emb"] re-reads and decompresses the
            # whole (N, 1152) array, so it must be read exactly once outside the loop.
            # And emb[i] is a view whose .base is that whole array, so keeping N of them
            # would pin N full copies -- .copy() gives each entry its own 1-D buffer.
            paths, emb = d["paths"], d["emb"]
            self._cache = {str(p): emb[i].copy() for i, p in enumerate(paths)}
            logger.info("Loaded %d cached image embeddings", len(self._cache))
        except Exception as e:
            logger.warning("Failed to load embedding cache: %s", e)

    def _save_cache(self):
        if not self.cache_path:
            return
        paths = list(self._cache.keys())
        emb = np.stack([self._cache[p] for p in paths]) if paths else np.zeros((0, 1))
        # np.savez auto-appends ".npz" unless the name already ends with it, so the
        # temp name must end in .npz for os.replace to find the written file.
        os.makedirs(os.path.dirname(os.path.abspath(self.cache_path)), exist_ok=True)
        tmp = self.cache_path + ".tmp.npz"
        np.savez(tmp, paths=np.array(paths), emb=emb)
        os.replace(tmp, self.cache_path)

    @staticmethod
    def frame_key(kf) -> str:
        """Cache key of a keyframe: its image-store frame_id (stable across store moves),
        prefixed by ``cache_ns`` for keyframes of another store, else its RGB path."""
        fid = getattr(kf, "frame_id", None)
        if fid is None:
            return str(kf.rgb_path)
        ns = getattr(kf, "cache_ns", None)
        return f"{ns}:f{int(fid)}" if ns else f"f{int(fid)}"

    def embed_images(
        self, paths: Sequence[str], keys: Optional[Sequence[str]] = None
    ) -> np.ndarray:
        """Embed ``paths``; ``keys`` (default: the paths) index the on-disk cache."""
        self._ensure()
        from PIL import Image

        torch = self._torch

        keys = list(paths) if keys is None else list(keys)
        key_path = dict(zip(keys, paths))
        missing = [
            k
            for k in dict.fromkeys(keys)
            if k not in self._cache and os.path.exists(key_path[k])
        ]
        for i in range(0, len(missing), self.batch_size):
            chunk = missing[i : i + self.batch_size]
            imgs = [Image.open(key_path[k]).convert("RGB") for k in chunk]
            with torch.no_grad():
                ii = self._proc(images=imgs, return_tensors="pt").to(self.device)
                e = self._feat(self._model.get_image_features(**ii)).float()
                e = e / e.norm(dim=-1, keepdim=True)
            for p, v in zip(chunk, e.cpu().numpy()):
                self._cache[p] = v
            logger.info(
                "embedded %d/%d new frames",
                min(i + len(chunk), len(missing)),
                len(missing),
            )
        if missing:
            self._save_cache()
        return np.stack([self._cache[k] for k in keys if k in self._cache])

    def embed_pil(self, images) -> np.ndarray:
        """Embed in-memory PIL images (detection crops). No disk cache — crops
        are query-specific and cheap (a handful per query, warm model)."""
        self._ensure()
        torch = self._torch
        out = []
        for i in range(0, len(images), self.batch_size):
            chunk = [im.convert("RGB") for im in images[i : i + self.batch_size]]
            with torch.no_grad():
                ii = self._proc(images=chunk, return_tensors="pt").to(self.device)
                e = self._feat(self._model.get_image_features(**ii)).float()
                e = e / e.norm(dim=-1, keepdim=True)
            out.append(e.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1))

    def score_crops(self, crops, texts: Sequence[str]) -> np.ndarray:
        """Cosine of each crop against each text: ``(n_crops, n_texts)``.

        The admission-evidence producer: row = one detection crop, columns = the
        target prompt followed by the negative-control prompts."""
        if not crops or not texts:
            return np.zeros((len(crops), len(texts)))
        return self.embed_pil(list(crops)) @ self.embed_texts(texts).T

    def embed_texts(self, prompts: Sequence[str]) -> np.ndarray:
        self._ensure()
        torch = self._torch
        with torch.no_grad():
            ti = self._proc(
                text=list(prompts), return_tensors="pt", padding="max_length"
            ).to(self.device)
            te = self._feat(self._model.get_text_features(**ti)).float()
            te = te / te.norm(dim=-1, keepdim=True)
        return te.cpu().numpy()

    def rank(self, keyframes, prompts: Sequence[str]) -> list:
        """Return [(keyframe, score)] sorted by descending max cosine to any prompt."""
        valid = [kf for kf in keyframes if kf.rgb_path and os.path.exists(kf.rgb_path)]
        if not valid or not prompts:
            return [(kf, 0.0) for kf in keyframes]
        img_emb = self.embed_images(
            [kf.rgb_path for kf in valid], keys=[self.frame_key(kf) for kf in valid]
        )
        txt_emb = self.embed_texts(prompts)
        score = (img_emb @ txt_emb.T).max(axis=1)
        order = np.argsort(-score)
        return [(valid[i], float(score[i])) for i in order]

    def select(
        self,
        keyframes,
        prompts: Sequence[str],
        top_k: int = 30,
        threshold: Optional[float] = None,
    ) -> list:
        """Return the keyframes most similar to any prompt (top-K, optional cutoff)."""
        if not keyframes or not prompts:
            return list(keyframes)
        valid = [kf for kf in keyframes if kf.rgb_path and os.path.exists(kf.rgb_path)]
        if not valid:
            logger.warning(
                "embed-filter: no readable keyframe rgb; passing all through"
            )
            return list(keyframes)

        img_emb = self.embed_images(
            [kf.rgb_path for kf in valid], keys=[self.frame_key(kf) for kf in valid]
        )
        txt_emb = self.embed_texts(prompts)
        sims = img_emb @ txt_emb.T  # (N, P)
        score = sims.max(axis=1)  # best-matching prompt per frame
        order = np.argsort(-score)

        selected = []
        for idx in order:
            if threshold is not None and score[idx] < threshold:
                break
            selected.append(valid[idx])
            if top_k and len(selected) >= top_k:
                break

        if selected:
            logger.info(
                "embed-filter: kept %d/%d frames (score range %.3f..%.3f)",
                len(selected),
                len(valid),
                float(score[order[0]]),
                float(score[order[len(selected) - 1]]),
            )
        return selected
