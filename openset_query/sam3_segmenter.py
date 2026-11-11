"""Thin wrapper around SAM3 / SAM3.1 open-vocabulary image segmentation.

Modeled on roman's ``fastsam_wrapper.py`` (mask + score per detection). Heavy
imports (torch, sam3) are deferred to :meth:`Sam3Segmenter._ensure_loaded` so the
package imports without a GPU or the SAM3 dependency present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Union

import numpy as np

from openset_query.config import Sam3FrontendConfig


@dataclass
class Detection:
    """A single SAM3 detection in image space."""

    label: str  # the text prompt that produced this detection
    mask: np.ndarray  # (H, W) bool
    box: np.ndarray  # (4,) xyxy in pixels
    score: float


class Sam3Segmenter:
    """Open-vocabulary text-prompted segmentation over single images."""

    def __init__(self, config: Sam3FrontendConfig):
        self.config = config
        self._model = None
        self._processor = None

    def _ensure_loaded(self):
        if self._processor is not None:
            return
        # Imported lazily: SAM3 pins numpy<2 and needs torch/CUDA.
        import torch
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        # TF32 + bf16 autocast match the SAM3 example notebooks for speed.
        if self.config.device.startswith("cuda") and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self._model = build_sam3_image_model(
            device=self.config.device, compile=getattr(self.config, "compile", False)
        )
        self._processor = Sam3Processor(
            self._model, confidence_threshold=self.config.conf_threshold
        )

    def set_conf_threshold(self, threshold: float) -> None:
        """Change the detection confidence threshold of the loaded model in place
        (single-image processor and batched postprocessor). The query server sets it
        per request under its GPU lock and restores it afterwards."""
        threshold = float(threshold)
        self.config.conf_threshold = threshold
        if self._processor is not None:
            self._processor.confidence_threshold = threshold
        if getattr(self, "_postprocessor", None) is not None:
            self._postprocessor.detection_threshold = threshold

    def warm_up(self, n: int = 3, size=(360, 640)):
        """Run a few dummy inferences so torch.compile compiles at startup, not on
        the first real query. Uses the keyframe resolution so shapes match."""
        import numpy as np
        self._ensure_loaded()
        dummy = (np.random.rand(size[0], size[1], 3) * 255).astype(np.uint8)
        for i in range(n):
            try:
                self.segment(dummy, ["object"])
            except Exception as e:  # pragma: no cover
                import logging
                logging.getLogger("openset_query.sam3_segmenter").warning(
                    "warm-up %d failed: %s", i, e)
                break

    def warm_up_batched(self, batch_size: int, size=(360, 640)):
        """Compile the batched path at a fixed batch size so the first real
        batched query doesn't stall on a torch.compile recompile."""
        import numpy as np
        self._ensure_batch()
        dummy = [(np.random.rand(size[0], size[1], 3) * 255).astype(np.uint8)
                 for _ in range(batch_size)]
        try:
            self.segment_batch(dummy, ["object"], fixed_batch=batch_size)
        except Exception as e:  # pragma: no cover
            import logging
            logging.getLogger("openset_query.sam3_segmenter").warning(
                "batched warm-up failed: %s", e)

    @staticmethod
    def _to_pil(image: Union["np.ndarray", "object"]):
        from PIL import Image

        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        return image  # assume already a PIL.Image

    def segment(
        self, image, prompts: Sequence[str]
    ) -> List[Detection]:
        """Run each text prompt against ``image`` and collect detections.

        Same result as the stock ``Sam3Processor`` path (:meth:`segment_reference`),
        split in two: :meth:`image_features` (the ViT trunk, ~85% of the time) and
        :meth:`segment_features` (neck + text + grounding). A caller holding cached
        features calls :meth:`segment_features` directly.

        Args:
            image: RGB ``np.ndarray`` (H, W, 3) or a PIL ``Image``.
            prompts: text queries (e.g. ``["chair", "fire extinguisher"]``).
        """
        if not self.supports_feature_cache():
            return self.segment_reference(image, prompts)
        feats, hw = self.image_features(image)
        return self.segment_features(feats, hw, prompts)

    # ------------------------------------------------------------------
    # Image-feature split (the SAM3 feature cache, see feature_cache.py).
    #
    # SAM3's image path is ViT trunk -> SimpleFPN neck (3 conv branches) ->
    # backbone_fpn; the text prompt + encoder/decoder/mask head then run per
    # prompt. The trunk output (1, 1024, 72, 72) is what we cache: every neck
    # branch starts with a conv that autocast runs in bf16, so storing the trunk
    # output in bf16 is lossless w.r.t. what the model consumes (10.6 MB/frame,
    # vs 56 MB for the neck outputs). The neck (~4 ms) and the image-independent
    # position encodings (computed once per shape) are recomputed per query.
    # ------------------------------------------------------------------

    def _neck(self):
        return self._model.backbone.vision_backbone

    def supports_feature_cache(self) -> bool:
        """True when the loaded SAM3 has the neck layout the split relies on."""
        self._ensure_loaded()
        ok = getattr(self, "_split_ok", None)
        if ok is None:
            neck = self._neck()
            ok = (hasattr(neck, "trunk") and hasattr(neck, "convs")
                  and hasattr(neck, "position_encoding")
                  and getattr(neck, "sam2_convs", None) is None
                  and self._model.inst_interactive_predictor is None)
            if not ok:
                import logging
                logging.getLogger("openset_query.sam3_segmenter").warning(
                    "SAM3 neck layout not recognised: feature cache disabled")
            self._split_ok = ok
        return ok

    def feature_version(self) -> str:
        """Short hash naming the cached features: the trunk weights (a few
        parameter tensors), the preprocessing (resolution, normalisation), the
        stored layout and eager vs torch.compile (a compiled trunk's bf16 output
        differs from eager by ~2-3% relative, so the two never share a cache:
        cached results stay identical to uncached ones in either mode)."""
        import hashlib

        import torch

        self._ensure_loaded()
        v = getattr(self, "_feat_version", None)
        if v is not None:
            return v
        h = hashlib.sha1()
        h.update(b"sam3-trunk-last-bf16-v1")
        h.update(b"compiled" if getattr(self.config, "compile", False) else b"eager")
        h.update(repr((self._processor.resolution, self._processor.transform)).encode())
        params = list(self._neck().trunk.parameters())
        for p in params[:2] + params[-2:]:
            h.update(p.detach().float().cpu().numpy().tobytes())
        h.update(str(sum(p.numel() for p in params)).encode())
        self._feat_version = h.hexdigest()[:12]
        return self._feat_version

    def _preprocess(self, image):
        """PIL / RGB array -> the (1, 3, R, R) normalised tensor + (h, w), exactly
        as ``Sam3Processor.set_image`` builds it."""
        from torchvision.transforms import v2

        pil = self._to_pil(image)
        w, h = pil.size
        x = v2.functional.to_image(pil).to(self._processor.device)
        return self._processor.transform(x).unsqueeze(0), (h, w)

    def image_features(self, image):
        """The ViT trunk output of ``image``: ``(features (1, C, h, w) bf16 on the
        device, (orig_h, orig_w))``. The expensive half of :meth:`segment`."""
        self._ensure_loaded()
        import torch

        with torch.inference_mode(), torch.autocast(
            self.config.device.split(":")[0], dtype=torch.bfloat16
        ):
            x, hw = self._preprocess(image)
            feats = self._neck().trunk(x)[-1].to(torch.bfloat16)
        return feats, hw

    def _backbone_out(self, feats):
        """Rebuild ``backbone_out`` (what ``backbone.forward_image`` returns) from
        cached trunk features: the neck convs + the (cached) position encodings,
        minus the ``scalp`` lowest-resolution levels."""
        import torch

        neck = self._neck()
        n = len(neck.convs) - max(0, int(getattr(self._model.backbone, "scalp", 0)))
        # The trunk emits channels-last; the conv kernels (and so the bf16
        # numerics) depend on the layout, so restore it for features that went
        # through disk (saved contiguous).
        feats = feats.to(self._processor.device).contiguous(memory_format=torch.channels_last)
        out, pos = [], []
        cache = self.__dict__.setdefault("_pos_cache", {})
        for i in range(n):
            x = neck.convs[i](feats)
            key = (i, tuple(x.shape), x.dtype, x.device)
            p = cache.get(key)
            if p is None:
                p = neck.position_encoding(x).to(x.dtype)
                cache[key] = p
            out.append(x)
            pos.append(p)
        return {"vision_features": out[-1], "vision_pos_enc": pos,
                "backbone_fpn": out, "sam2_backbone_out": None}

    def _text_outputs(self, prompt: str):
        """``backbone.forward_text([prompt])``, memoised per prompt (it is
        image-independent, so a query encodes its prompt once, not per frame)."""
        cache = self.__dict__.setdefault("_text_cache", {})
        out = cache.get(prompt)
        if out is None:
            out = self._model.backbone.forward_text([prompt], device=self._processor.device)
            if len(cache) >= 64:
                cache.pop(next(iter(cache)))
            cache[prompt] = out
        return out

    def segment_features(self, feats, hw, prompts: Sequence[str]) -> List[Detection]:
        """Detections of ``prompts`` given cached trunk ``feats`` of an image of
        size ``hw`` = (h, w). Matches :meth:`segment_reference` on that image."""
        self._ensure_loaded()
        import torch

        detections: List[Detection] = []
        with torch.inference_mode(), torch.autocast(
            self.config.device.split(":")[0], dtype=torch.bfloat16
        ):
            backbone_out = self._backbone_out(feats)
            for prompt in prompts:
                state = {"original_height": int(hw[0]), "original_width": int(hw[1]),
                         "backbone_out": dict(backbone_out, **self._text_outputs(prompt)),
                         "geometric_prompt": self._model._get_dummy_prompt()}
                state = self._processor._forward_grounding(state)
                detections.extend(self._collect(state, prompt))
        return detections

    def _collect(self, state, prompt) -> List[Detection]:
        masks, boxes, scores = state.get("masks"), state.get("boxes"), state.get("scores")
        if masks is None or len(scores) == 0:
            return []
        masks_np = self._as_numpy(masks)
        boxes_np = self._as_numpy(boxes)
        scores_np = self._as_numpy(scores)
        out = []
        for i in range(masks_np.shape[0]):
            # SAM3 masks are (N, 1, H, W); squeeze the channel dim.
            m = np.squeeze(masks_np[i]).astype(bool)
            out.append(Detection(label=prompt, mask=m, box=boxes_np[i].astype(np.float64),
                                 score=float(scores_np[i])))
        return out

    def segment_reference(
        self, image, prompts: Sequence[str]
    ) -> List[Detection]:
        """The stock SAM3 path (``Sam3Processor.set_image`` + ``set_text_prompt``),
        kept to verify that :meth:`segment` / the feature cache match it.

        Args:
            image: RGB ``np.ndarray`` (H, W, 3) or a PIL ``Image``.
            prompts: text queries (e.g. ``["chair", "fire extinguisher"]``).
        """
        self._ensure_loaded()
        import torch

        pil = self._to_pil(image)
        detections: List[Detection] = []

        with torch.inference_mode(), torch.autocast(
            self.config.device.split(":")[0], dtype=torch.bfloat16
        ):
            state = self._processor.set_image(pil)
            for prompt in prompts:
                self._processor.reset_all_prompts(state)
                state = self._processor.set_text_prompt(state=state, prompt=prompt)
                masks = state.get("masks")
                boxes = state.get("boxes")
                scores = state.get("scores")
                if masks is None or len(scores) == 0:
                    continue
                masks_np = self._as_numpy(masks)
                boxes_np = self._as_numpy(boxes)
                scores_np = self._as_numpy(scores)
                for i in range(masks_np.shape[0]):
                    m = masks_np[i]
                    # SAM3 masks are (N, 1, H, W); squeeze the channel dim.
                    m = np.squeeze(m).astype(bool)
                    detections.append(
                        Detection(
                            label=prompt,
                            mask=m,
                            box=boxes_np[i].astype(np.float64),
                            score=float(scores_np[i]),
                        )
                    )
        return detections

    @staticmethod
    def _as_numpy(t) -> np.ndarray:
        if hasattr(t, "detach"):
            return t.detach().to("cpu").float().numpy()
        return np.asarray(t)

    # ------------------------------------------------------------------
    # Batched inference (one forward over many images) via SAM3's Datapoint API.
    # The simple Sam3Processor only batches the backbone, not grounding, so for a
    # true batched forward we use sam3.train's Datapoint/collate/model/postprocessor.
    # ------------------------------------------------------------------

    def _ensure_batch(self):
        if getattr(self, "_batch_ready", False):
            return
        self._ensure_loaded()
        from sam3.train.transforms.basic_for_api import (
            ComposeAPI, RandomResizeAPI, ToTensorAPI, NormalizeAPI)
        from sam3.eval.postprocessors import PostProcessImage

        self._transform = ComposeAPI(transforms=[
            RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self._postprocessor = PostProcessImage(
            max_dets_per_img=-1, iou_type="segm",
            use_original_sizes_box=True, use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            detection_threshold=self.config.conf_threshold, to_cpu=False)
        self._batch_ready = True

    def segment_batch(self, images, prompts: Sequence[str], fixed_batch=None) -> List[List[Detection]]:
        """Run all prompts over all images, returning per-image Detection lists.

        If ``fixed_batch`` is set, images are processed in chunks of exactly that
        size — the last chunk padded with black frames — so every forward has the
        SAME batch shape. This keeps torch.compile from recompiling per batch size
        (which otherwise stalls ~40 s on each new size). Padding outputs are dropped.
        """
        pil_images = [self._to_pil(im) for im in images]
        if not fixed_batch or len(pil_images) == fixed_batch:
            return self._segment_one_batch(pil_images, prompts)

        from PIL import Image
        w, h = pil_images[0].size
        black = Image.new("RGB", (w, h))
        out: List[List[Detection]] = []
        for i in range(0, len(pil_images), fixed_batch):
            chunk = pil_images[i:i + fixed_batch]
            n_real = len(chunk)
            if n_real < fixed_batch:  # pad to the fixed size
                chunk = chunk + [black] * (fixed_batch - n_real)
            res = self._segment_one_batch(chunk, prompts)
            out.extend(res[:n_real])  # drop padding-frame results
        return out

    def _segment_one_batch(self, pil_images, prompts: Sequence[str]) -> List[List[Detection]]:
        """One batched forward over a list of PIL images (fixed shape)."""
        self._ensure_batch()
        import torch
        from sam3.train.data.collator import collate_fn_api as collate
        from sam3.model.utils.misc import copy_data_to_device
        from sam3.train.data.sam3_image_dataset import (
            InferenceMetadata, FindQueryLoaded, Image as SAMImage, Datapoint)

        datapoints = []
        id_map = {}  # query id -> (image_index, prompt)
        qid = 1
        for img_idx, pil in enumerate(pil_images):
            w, h = pil.size
            dp = Datapoint(find_queries=[], images=[SAMImage(data=pil, objects=[], size=[h, w])])
            for prompt in prompts:
                dp.find_queries.append(FindQueryLoaded(
                    query_text=prompt, image_id=0, object_ids_output=[],
                    is_exhaustive=True, query_processing_order=0,
                    inference_metadata=InferenceMetadata(
                        coco_image_id=qid, original_image_id=qid, original_category_id=1,
                        # postprocessor reads target size as [h, w] (img_h, img_w =
                        # unbind). The SAM3 notebook's [w, h] transposes non-square images.
                        original_size=[h, w], object_id=0, frame_index=0)))
                id_map[qid] = (img_idx, prompt)
                qid += 1
            datapoints.append(self._transform(dp))

        batch = collate(datapoints, dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, torch.device(self.config.device), non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            self.config.device.split(":")[0], dtype=torch.bfloat16):
            output = self._model(batch)
        results = self._postprocessor.process_results(output, batch.find_metadatas)

        per_image: List[List[Detection]] = [[] for _ in pil_images]
        for q, (img_idx, prompt) in id_map.items():
            r = results.get(q)
            if r is None:
                continue
            scores = self._as_numpy(r["scores"])
            masks = self._as_numpy(r["masks"]) if not isinstance(r["masks"], list) else r["masks"]
            boxes = self._as_numpy(r["boxes"])
            for k in range(len(scores)):
                m = masks[k]
                m = np.squeeze(self._as_numpy(m)).astype(bool)
                per_image[img_idx].append(Detection(
                    label=prompt, mask=m, box=boxes[k].astype(np.float64), score=float(scores[k])))
        return per_image
