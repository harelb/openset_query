import io
import json
import urllib.error
import pytest
from openset_query import query_client as qc


class _Resp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False


def test_sam3_query_success(monkeypatch):
    payload = {"prompt": "chair", "objects": 2, "symbols": ["O1", "O2"], "elapsed_s": 1.3}
    monkeypatch.setattr(qc.urllib.request, "urlopen",
                        lambda req, timeout=0: _Resp(json.dumps(payload).encode()))
    out = qc.sam3_query("chair", base_url="http://x:1")
    assert out["symbols"] == ["O1", "O2"]


def test_sam3_query_unreachable(monkeypatch):
    def _raise(req, timeout=0): raise urllib.error.URLError("refused")
    monkeypatch.setattr(qc.urllib.request, "urlopen", _raise)
    with pytest.raises(qc.Sam3Unavailable):
        qc.sam3_query("chair", base_url="http://x:1")


def test_sam3_query_server_error(monkeypatch):
    monkeypatch.setattr(qc.urllib.request, "urlopen",
                        lambda req, timeout=0: _Resp(json.dumps({"error": "boom"}).encode()))
    with pytest.raises(qc.Sam3QueryError):
        qc.sam3_query("chair", base_url="http://x:1")
