import sys

import pytest

from adam_tts.errors import AlignError, DependencyMissing
from adam_tts.workers import client as client_mod

FAKE = '''
import sys
sys.path.insert(0, {wd!r})
from _proto import serve
print("noise on stdout must not break the protocol")
def echo(req): print("more noise"); return {{"x": req["x"] * 2}}
def boom(req): raise ValueError("bad")
def imp(req): import definitely_not_a_module  # noqa
serve({{"echo": echo, "boom": boom, "imp": imp}})
'''


@pytest.fixture
def fake_worker(tmp_path, monkeypatch):
    from adam_tts.workers import WORKER_DIR

    d = tmp_path / "w"
    d.mkdir()
    (d / "fake_worker.py").write_text(FAKE.format(wd=str(WORKER_DIR)))
    monkeypatch.setattr(client_mod, "WORKER_DIR", d)
    return d


def test_protocol_roundtrip_and_errors(fake_worker):
    c = client_mod.WorkerClient(sys.executable, "fake_worker.py", error_cls=AlignError, label="fake")
    try:
        assert c.request("echo", x=21)["x"] == 42
        with pytest.raises(AlignError, match="bad"):
            c.request("boom")
        with pytest.raises(DependencyMissing):
            c.request("imp")
        assert c.request("echo", x=1)["x"] == 2
    finally:
        c.close()
