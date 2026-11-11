"""SAM3 image-feature cache: per-frame ViT trunk outputs, computed once, reused per query.

SAM3 spends ~85% of a frame's time in the image trunk (``Sam3Segmenter.image_features``);
the text prompt, neck and grounding (``segment_features``) are cheap. The keyframe pool
does not change between queries, so the trunk output of every pool frame is cached:

* **GPU tier** (LRU, byte budget): features ready for ``segment_features``;
* **CPU tier** (LRU, byte budget): features evicted from the GPU;
* **disk**: one safetensors file per frame under
  ``<image_store>/cache/sam3_feats_<version>/`` (``version`` = trunk weights +
  preprocessing, see ``Sam3Segmenter.feature_version``), written by a background
  writer so a query never waits on the disk.

:class:`FeatureCacheBuilder` fills the cache in the background (server startup, new
live-mirror frames on ``/reload``) and yields the GPU to queries through a
:class:`GpuGate`. A query that hits an uncached frame computes it on the spot and
caches it, so results never depend on the cache state: cached features reproduce the
uncached detections exactly (bf16 trunk output, the dtype the neck consumes).

Torch is only touched through a *codec* (:class:`TorchCodec`), so the cache logic is
unit-tested with plain numpy arrays.
"""

from __future__ import annotations

import collections
import hashlib
import logging
import os
import queue
import re
import threading
import time
from contextlib import contextmanager
from typing import Callable, Iterable, Optional

logger = logging.getLogger("openset_query.feature_cache")

GB = 1024 ** 3


class GpuGate:
    """The GPU lock shared by queries (foreground) and the cache builder
    (background): a waiting query always goes before the builder's next frame.

    Used as ``with gate:`` (foreground), like the ``threading.Lock`` it replaces.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._cv = threading.Condition()
        self._waiting = 0
        self._fg_active = 0

    def __enter__(self):
        with self._cv:
            self._waiting += 1
        try:
            self._lock.acquire()
        finally:
            with self._cv:
                self._waiting -= 1
                self._fg_active += 1
                self._cv.notify_all()
        return self

    def __exit__(self, *exc):
        with self._cv:
            self._fg_active -= 1
        self._lock.release()
        return False

    @contextmanager
    def background(self):
        """Hold the GPU for one unit of background work, after any waiting query."""
        with self._cv:
            while self._waiting:
                self._cv.wait(0.05)
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()

    def locked(self) -> bool:
        """A query holds (or waits for) the GPU (background work does not count)."""
        return self._fg_active > 0 or self._waiting > 0


class TorchCodec:
    """Feature tensors <-> devices / safetensors files."""

    def __init__(self, device: str = "cuda"):
        self.device = device

    @staticmethod
    def nbytes(feats) -> int:
        return int(feats.numel() * feats.element_size())

    def to_gpu(self, feats):
        return feats.to(self.device, non_blocking=True)

    @staticmethod
    def to_cpu(feats):
        return feats.detach().to("cpu")

    @staticmethod
    def save(path: str, feats, hw) -> None:
        from safetensors.torch import save_file

        save_file({"feats": feats.detach().to("cpu").contiguous()}, path,
                  metadata={"h": str(int(hw[0])), "w": str(int(hw[1]))})

    def load(self, path: str, to_gpu: bool = True):
        from safetensors import safe_open

        dev = self.device if to_gpu else "cpu"
        with safe_open(path, framework="pt", device=dev) as f:
            meta = f.metadata() or {}
            feats = f.get_tensor("feats")
        return feats, (int(meta["h"]), int(meta["w"]))


def _safe_name(key: str) -> str:
    """File name of a frame key: readable, collision-free (keys hold ':' and '/')."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", key)[-80:]
    return f"{safe}-{hashlib.sha1(key.encode()).hexdigest()[:8]}.safetensors"


class FeatureCache:
    """Three-tier (GPU / CPU / disk) LRU of per-frame SAM3 trunk features."""

    def __init__(self, cache_dir: Optional[str], codec, gpu_budget_bytes: int = 4 * GB,
                 cpu_budget_bytes: int = 4 * GB, max_disk_bytes: int = 64 * GB,
                 write_async: bool = True):
        self.cache_dir = cache_dir
        self.codec = codec
        self.gpu_budget = max(0, int(gpu_budget_bytes))
        self.cpu_budget = max(0, int(cpu_budget_bytes))
        self.max_disk_bytes = int(max_disk_bytes)
        self._gpu: "collections.OrderedDict[str, tuple]" = collections.OrderedDict()
        self._cpu: "collections.OrderedDict[str, tuple]" = collections.OrderedDict()
        self._gpu_bytes = self._cpu_bytes = 0
        self._lock = threading.RLock()
        self._on_disk: set = set()
        self._disk_bytes = 0
        self._disk_ok = bool(cache_dir)
        self._pending_writes: set = set()
        if self._disk_ok:
            try:
                os.makedirs(cache_dir, exist_ok=True)
                for name in os.listdir(cache_dir):
                    if name.endswith(".safetensors"):
                        self._on_disk.add(name)
                        self._disk_bytes += os.path.getsize(os.path.join(cache_dir, name))
            except OSError as e:
                logger.warning("feature cache dir %s unusable (%s): memory only", cache_dir, e)
                self._disk_ok = False
        self._writes: "queue.Queue" = queue.Queue()
        self._writer = None
        if self._disk_ok and write_async:
            self._writer = threading.Thread(target=self._write_loop, name="sam3-feat-writer",
                                            daemon=True)
            self._writer.start()

    # -- lookup ---------------------------------------------------------------

    def path(self, key: str) -> Optional[str]:
        return os.path.join(self.cache_dir, _safe_name(key)) if self.cache_dir else None

    def on_disk(self, key: str) -> bool:
        with self._lock:
            return _safe_name(key) in self._on_disk

    def has(self, key: str) -> bool:
        with self._lock:
            return key in self._gpu or key in self._cpu or key in self._pending_writes \
                or _safe_name(key) in self._on_disk

    def tier(self, key: str) -> Optional[str]:
        with self._lock:
            if key in self._gpu:
                return "gpu"
            if key in self._cpu:
                return "cpu"
            if _safe_name(key) in self._on_disk:
                return "disk"
        return None

    def get(self, key: str):
        """``(feats, hw, tier)`` with ``feats`` on the GPU (promoted), or None.
        ``tier`` is where it was found: "gpu", "cpu" or "disk"."""
        with self._lock:
            hit = self._gpu.get(key)
            if hit is not None:
                self._gpu.move_to_end(key)
                return hit[0], hit[1], "gpu"
            hit = self._cpu.pop(key, None)
            if hit is not None:
                self._cpu_bytes -= hit[2]
                feats = self.codec.to_gpu(hit[0])
                self._insert_gpu(key, feats, hit[1])
                return feats, hit[1], "cpu"
            on_disk = _safe_name(key) in self._on_disk
        if not on_disk:
            return None
        try:
            feats, hw = self.codec.load(self.path(key), to_gpu=True)
        except Exception as e:
            logger.warning("feature cache: dropping unreadable %s (%s)", self.path(key), e)
            with self._lock:
                self._on_disk.discard(_safe_name(key))
            return None
        with self._lock:
            self._insert_gpu(key, feats, hw)
        return feats, hw, "disk"

    # -- insert ---------------------------------------------------------------

    def put(self, key: str, feats, hw, persist: bool = True) -> None:
        """Cache ``feats`` (on the GPU) for ``key``; also write it to disk unless
        it is there already (in the background when the writer runs)."""
        hw = (int(hw[0]), int(hw[1]))
        with self._lock:
            self._insert_gpu(key, feats, hw)
            want_disk = (persist and self._disk_ok and _safe_name(key) not in self._on_disk
                         and key not in self._pending_writes
                         and self._disk_bytes < self.max_disk_bytes)
            if want_disk:
                self._pending_writes.add(key)
        if want_disk:
            if self._writer is not None:
                self._writes.put((key, self.codec.to_cpu(feats), hw))
            else:
                self._write(key, feats, hw)

    def _insert_gpu(self, key, feats, hw):
        n = self.codec.nbytes(feats)
        old = self._gpu.pop(key, None)
        if old is not None:
            self._gpu_bytes -= old[2]
        if n > self.gpu_budget:
            self._insert_cpu(key, self.codec.to_cpu(feats), hw)
            return
        self._gpu[key] = (feats, hw, n)
        self._gpu_bytes += n
        while self._gpu_bytes > self.gpu_budget and self._gpu:
            k, (f, h, b) = self._gpu.popitem(last=False)
            self._gpu_bytes -= b
            self._insert_cpu(k, self.codec.to_cpu(f), h)

    def _insert_cpu(self, key, feats, hw):
        n = self.codec.nbytes(feats)
        if n > self.cpu_budget:
            return  # disk (if written) still has it
        old = self._cpu.pop(key, None)
        if old is not None:
            self._cpu_bytes -= old[2]
        self._cpu[key] = (feats, hw, n)
        self._cpu_bytes += n
        while self._cpu_bytes > self.cpu_budget and self._cpu:
            _, (_, _, b) = self._cpu.popitem(last=False)
            self._cpu_bytes -= b

    def preload(self, keys: Iterable[str], stop: Optional[Callable[[], bool]] = None) -> int:
        """Load on-disk features of ``keys`` into memory while the GPU tier has room."""
        n = 0
        for key in keys:
            if stop is not None and stop():
                break
            with self._lock:
                if key in self._gpu or key in self._cpu:
                    continue
                full = self._gpu and self._gpu_bytes + next(iter(self._gpu.values()))[2] > self.gpu_budget
            if full:
                break
            if self.get(key) is not None:
                n += 1
        return n

    # -- disk -----------------------------------------------------------------

    def _write(self, key, feats, hw):
        path = self.path(key)
        tmp = path + ".tmp"
        try:
            self.codec.save(tmp, feats, hw)
            os.replace(tmp, path)
            size = os.path.getsize(path)
            with self._lock:
                self._on_disk.add(_safe_name(key))
                self._disk_bytes += size
        except Exception as e:
            logger.warning("feature cache: cannot write %s (%s); disk tier disabled", path, e)
            self._disk_ok = False
            try:
                os.remove(tmp)
            except OSError:
                pass
        finally:
            with self._lock:
                self._pending_writes.discard(key)

    def _write_loop(self):
        while True:
            item = self._writes.get()
            try:
                if item is None:
                    return
                self._write(*item)
            finally:
                self._writes.task_done()

    def flush(self) -> None:
        """Wait for the background writes queued so far."""
        if self._writer is not None:
            self._writes.join()

    def close(self) -> None:
        if self._writer is not None:
            self.flush()
            self._writes.put(None)
            self._writer.join(timeout=5)
            self._writer = None

    def stats(self) -> dict:
        with self._lock:
            return {"gpu": len(self._gpu), "cpu": len(self._cpu),
                    "disk": len(self._on_disk),
                    "gpu_gb": round(self._gpu_bytes / GB, 2),
                    "cpu_gb": round(self._cpu_bytes / GB, 2),
                    "disk_gb": round(self._disk_bytes / GB, 2),
                    "gpu_budget_gb": round(self.gpu_budget / GB, 2),
                    "cpu_budget_gb": round(self.cpu_budget / GB, 2),
                    "dir": self.cache_dir}


class CacheStats:
    """Per-query hit counters (by tier) for the response timings."""

    def __init__(self):
        self.counts = collections.Counter()
        self.feat_s = 0.0  # time spent getting features (cache read or trunk)

    def add(self, tier: str, dt: float = 0.0):
        self.counts[tier] += 1
        self.feat_s += dt

    def to_json(self) -> dict:
        n = sum(self.counts.values())
        hits = n - self.counts.get("miss", 0)
        return {"frames": n, "hits": hits, "hit_rate": round(hits / n, 3) if n else None,
                **{k: self.counts.get(k, 0) for k in ("gpu", "cpu", "disk", "miss")},
                "feat_s": round(self.feat_s, 3)}


class CachedSegmenter:
    """``Sam3Segmenter`` view that takes the image features from a
    :class:`FeatureCache` (computing + caching on a miss).

    Detection code calls ``segment_keyframe(kf, rgb, prompts)`` when the segmenter
    has it (see ``object_discovery._segment_kf``); everything else is delegated.
    """

    def __init__(self, seg, cache: FeatureCache, key_fn: Callable):
        self._seg, self.cache, self._key = seg, cache, key_fn
        self.stats = CacheStats()

    def reset_stats(self) -> CacheStats:
        s, self.stats = self.stats, CacheStats()
        return s

    def features(self, kf, rgb=None):
        """``(feats, hw, tier)`` of ``kf``; ``tier`` "miss" when just computed."""
        key = self._key(kf)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        if rgb is None:
            from openset_query import keyframe_store as ks

            rgb = ks.read_rgb(kf)
        feats, hw = self._seg.image_features(rgb)
        self.cache.put(key, feats, hw)
        return feats, hw, "miss"

    def segment_keyframe(self, kf, rgb, prompts):
        t = time.perf_counter()
        feats, hw, tier = self.features(kf, rgb)
        self.stats.add(tier, time.perf_counter() - t)
        return self._seg.segment_features(feats, hw, prompts)

    def __getattr__(self, name):
        return getattr(self._seg, name)


class FeatureCacheBuilder:
    """Background thread computing the features of queued keyframes.

    Each frame: read the RGB (no lock), then compute + cache it under
    ``gate.background()`` so queries are served between frames. After the queue
    drains, on-disk features are preloaded into the GPU tier while it has room.
    """

    def __init__(self, cached: CachedSegmenter, gate: GpuGate, key_fn: Callable,
                 read_rgb: Optional[Callable] = None):
        self.cached, self.gate, self._key = cached, gate, key_fn
        if read_rgb is None:
            from openset_query import keyframe_store as ks

            read_rgb = ks.read_rgb
        self._read_rgb = read_rgb
        self._q: "collections.deque" = collections.deque()
        self._preload: list = []
        self._cv = threading.Condition()
        self._thread = None
        self._active = False
        self._stop = False
        self.built = 0
        self.failed = 0
        self.build_s = 0.0
        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None

    def enqueue(self, keyframes) -> int:
        """Queue the frames not cached yet (memory or disk) and preload the cached
        ones; returns how many frames will be computed."""
        cache = self.cached.cache
        todo, warm = [], []
        for kf in keyframes:
            k = self._key(kf)
            (warm if cache.has(k) else todo).append(kf)
        start = False
        with self._cv:
            queued = {self._key(kf) for kf in self._q}
            todo = [kf for kf in todo if self._key(kf) not in queued]
            self._q.extend(todo)
            self._preload = [self._key(kf) for kf in warm]
            if (todo or warm) and not self._active:
                self._active, start = True, True
                self.finished_at = None
            self._cv.notify_all()
        if start:
            self._thread = threading.Thread(target=self._run, name="sam3-feat-builder",
                                            daemon=True)
            self._thread.start()
        return len(todo)

    def pending(self) -> int:
        with self._cv:
            return len(self._q)

    def running(self) -> bool:
        return self._active

    def stop(self):
        self._stop = True

    def wait(self, timeout: Optional[float] = None) -> bool:
        """Block until the builder is idle (tests / the precompute CLI)."""
        t0 = time.time()
        while self._active:
            if timeout is not None and time.time() - t0 > timeout:
                return False
            time.sleep(0.02)
        return True

    def _run(self):
        if self.started_at is None:
            self.started_at = time.time()
        while not self._stop:
            keys = None
            with self._cv:
                kf = self._q.popleft() if self._q else None
                if kf is None:
                    keys, self._preload = self._preload, []
                    if not keys:
                        self._active = False
                        self.finished_at = time.time()
                        logger.info("feature cache: built %d frames (%.1f s, %d failed) | %s",
                                    self.built, self.build_s, self.failed,
                                    self.cached.cache.stats())
                        return
            if kf is None:
                n = self.cached.cache.preload(keys, stop=lambda: self._stop or bool(self._q))
                if n:
                    logger.info("feature cache: preloaded %d frames from disk", n)
                continue
            self._build_one(kf)
        with self._cv:
            self._active = False

    def _build_one(self, kf):
        key = self._key(kf)
        if self.cached.cache.has(key):
            return
        try:
            rgb = self._read_rgb(kf)
        except Exception as e:
            logger.warning("feature cache: cannot read %s: %s", getattr(kf, "rgb_path", kf), e)
            self.failed += 1
            return
        t = time.perf_counter()
        try:
            with self.gate.background():
                if not self.cached.cache.has(key):
                    self.cached.features(kf, rgb)
            self.built += 1
        except Exception:
            logger.exception("feature cache: computing %s failed", key)
            self.failed += 1
        self.build_s += time.perf_counter() - t

    def status(self) -> dict:
        return {"building": self.running() and self.pending() > 0, "queued": self.pending(),
                "built": self.built, "failed": self.failed,
                "build_s": round(self.build_s, 1),
                "per_frame_s": round(self.build_s / self.built, 3) if self.built else None}
