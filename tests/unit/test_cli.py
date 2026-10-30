import json
import wave

import pytest

from adam_tts import __version__
from adam_tts.backends.base import CheckResult
from adam_tts.backends.edge import EdgeBackend
from adam_tts.backends.mimo import MimoBackend
from adam_tts.cli import main
from conftest import FAKE_KEY, FakeBackend


def run(capsys, *argv):
    code = main(list(argv))
    out, err = capsys.readouterr()
    return code, out, err


def j(out):
    lines = [l for l in out.splitlines() if l.strip()]
    assert len(lines) == 1, out  # exactly one JSON object on stdout
    return json.loads(lines[0])


def test_version(capsys):
    assert run(capsys, "--version")[1].strip() == f"adam_tts {__version__}"
    code, out, _ = run(capsys, "--version", "--json")
    assert code == 0 and j(out) == {"ok": True, "version": __version__}


def test_no_command_is_usage_error(capsys):
    code, out, _ = run(capsys, "--json")
    assert code == 2 and j(out)["error"]["type"] == "UsageError"


def test_argparse_error_is_json(capsys):
    code, out, _ = run(capsys, "say", "x", "--json")  # missing -o
    assert code == 2 and j(out)["ok"] is False


def test_say_with_fake_backend(capsys, tmp_path):
    out_wav = tmp_path / "o.wav"
    code, out, _ = run(capsys, "say", "你好世界。", "-o", str(out_wav), "--backend", "fake", "--json")
    r = j(out)
    assert code == 0 and r["ok"] and r["backend"] == "fake" and r["voice"] == "fv1" and r["preset"] is None
    assert out_wav.exists() and r["sample_rate"] == 24000 and r["chunks"] == 1


def test_say_uses_config_default_preset(capsys, tmp_path, fake_config):
    code, out, _ = run(capsys, "say", "你好。", "-o", str(tmp_path / "o.wav"), "--json")
    r = j(out)
    assert code == 0 and r["preset"] == "fk" and r["voice"] == "fv2"
    req = FakeBackend.calls[0]
    assert req.style == "平静" and req.prefix == "(平静)"


def test_say_chunks_long_text_and_concats(capsys, tmp_path):
    text = "这是第一句话的内容。" * 5  # 50 chars, fake max 20
    out_wav = tmp_path / "o.wav"
    code, out, _ = run(capsys, "say", text, "-o", str(out_wav), "--backend", "fake", "--json")
    r = j(out)
    assert code == 0 and r["chunks"] == 3
    assert "".join(c.text for c in FakeBackend.calls) == text
    with wave.open(str(out_wav)) as w:
        assert abs(w.getnframes() / w.getframerate() - 0.5) < 0.01


def test_say_text_file(capsys, tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("文件里的文本。\n", encoding="utf-8")
    code, _, _ = run(capsys, "say", "--text-file", str(f), "-o", str(tmp_path / "o.wav"), "--backend", "fake")
    assert code == 0 and FakeBackend.calls[0].text == "文件里的文本。"


def test_say_text_and_file_conflict(capsys, tmp_path):
    code, _, _ = run(capsys, "say", "x", "--text-file", "f", "-o", str(tmp_path / "o.wav"))
    assert code == 2


def test_say_non_wav_output(capsys, tmp_path):
    assert run(capsys, "say", "x", "-o", str(tmp_path / "o.mp3"), "--backend", "fake")[0] == 2


def test_unsupported_param_exit_4_before_request(capsys, tmp_path):
    code, out, _ = run(capsys, "say", "x", "-o", str(tmp_path / "o.wav"), "--backend", "fake",
                       "--speed", "1.2", "--json")
    assert code == 4 and j(out)["error"]["type"] == "UnsupportedParam" and FakeBackend.calls == []


def test_mimo_speed_errors_before_network(capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(MimoBackend, "synthesize", lambda *a: pytest.fail("must not call backend"))
    assert run(capsys, "say", "x", "-o", str(tmp_path / "o.wav"), "--speed", "1.1")[0] == 4


def test_preset_backend_conflict_exit_2(capsys, tmp_path):
    assert run(capsys, "say", "x", "-o", str(tmp_path / "o.wav"), "--preset", "draft", "--backend", "edge")[0] == 2


def test_backend_error_exit_5(capsys, tmp_path):
    FakeBackend.fail = True
    code, out, _ = run(capsys, "say", "x", "-o", str(tmp_path / "o.wav"), "--backend", "fake", "--json")
    assert code == 5 and j(out)["error"]["type"] == "BackendError"


def test_mimo_missing_key_exit_3(capsys, tmp_path):
    code, out, _ = run(capsys, "say", "x", "-o", str(tmp_path / "o.wav"), "--json")
    assert code == 3 and "MIMO_API_KEY" in j(out)["error"]["message"]


@pytest.mark.parametrize("backend", ["doubao"])
def test_stub_backends_exit_8(capsys, tmp_path, backend):
    code, out, _ = run(capsys, "say", "x", "-o", str(tmp_path / "o.wav"), "--backend", backend, "--voice", "v",
                       "--json")
    assert code == 8 and j(out)["error"]["type"] == "NotImplementedBackend"
    code, out, _ = run(capsys, "voice", "list", "--backend", backend, "--json")
    assert code == 8


def test_voice_list(capsys):
    code, out, _ = run(capsys, "voice", "list", "--backend", "mimo", "--lang", "zh", "--json")
    r = j(out)
    assert code == 0 and {v["name"] for v in r["voices"]} >= {"苏打", "冰糖", "白桦", "茉莉"}
    assert "Mia" not in {v["name"] for v in r["voices"]}
    code, out, _ = run(capsys, "voice", "list", "--backend", "fake")
    assert code == 0 and "fv1" in out


def test_voice_without_subcommand(capsys):
    assert run(capsys, "voice", "--json")[0] == 2


def _patch_checks(monkeypatch, mimo_status="ok"):
    monkeypatch.setattr(MimoBackend, "check", lambda self, deep=False: [
        CheckResult("mimo.key", mimo_status, "MIMO_API_KEY " + self.secrets.source("MIMO_API_KEY"), 3)])
    monkeypatch.setattr(EdgeBackend, "check", lambda self, deep=False: [CheckResult("edge.ok", "ok", "fine")])


def test_doctor_healthy_and_never_prints_key(capsys, monkeypatch):
    monkeypatch.setenv("MIMO_API_KEY", FAKE_KEY)
    _patch_checks(monkeypatch)
    code, out, err = run(capsys, "doctor", "--json")
    r = j(out)
    assert code == 0 and r["ok"] and r["healthy"]
    assert r["secrets"]["MIMO_API_KEY"] == "env" and r["secrets"]["DOUBAO_API_KEY"] == "missing"
    names = {c["name"]: c for c in r["checks"]}
    assert names["doubao.implemented"]["status"] == "skip" and not names["doubao.implemented"]["required"]
    assert not names["qwen.model"]["required"]
    assert names["mimo.key"]["required"] and not names["doubao.key"]["required"]
    assert FAKE_KEY not in out + err
    code, out, err = run(capsys, "doctor")
    assert FAKE_KEY not in out + err and "healthy" in out


def test_doctor_required_failure_exit_3(capsys, monkeypatch):
    _patch_checks(monkeypatch, mimo_status="fail")
    code, out, _ = run(capsys, "doctor", "--json")
    assert code == 3 and j(out)["ok"] is False


def test_doctor_explicit_stub_backend_fails(capsys):
    code, out, _ = run(capsys, "doctor", "--backend", "qwen", "--json")
    assert code == 8 and j(out)["healthy"] is False


def test_doctor_fake_backend(capsys):
    code, out, _ = run(capsys, "doctor", "--backend", "fake", "--json")
    assert code == 0


def test_dotenv_used_by_cli(capsys, tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(f"MIMO_API_KEY={FAKE_KEY}\n")
    _patch_checks(monkeypatch)
    code, out, err = run(capsys, "doctor", "--json")
    assert j(out)["secrets"]["MIMO_API_KEY"] == ".env" and FAKE_KEY not in out + err


def test_missing_explicit_env_file(capsys, tmp_path):
    assert run(capsys, "doctor", "--env-file", str(tmp_path / "nope"), "--json")[0] == 2
