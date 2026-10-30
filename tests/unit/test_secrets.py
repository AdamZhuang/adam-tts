from adam_tts.secrets import Secrets
from conftest import FAKE_KEY


def test_env_wins_over_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("MIMO_API_KEY=from-file\n")
    monkeypatch.setenv("MIMO_API_KEY", "from-env")
    s = Secrets()
    assert s.get("MIMO_API_KEY") == "from-env" and s.source("MIMO_API_KEY") == "env"


def test_dotenv_supplements_without_touching_environ(tmp_path, monkeypatch):
    import os

    (tmp_path / ".env").write_text("MIMO_API_KEY=from-file\n")
    s = Secrets()
    assert s.get("MIMO_API_KEY") == "from-file" and s.source("MIMO_API_KEY") == ".env"
    assert "MIMO_API_KEY" not in os.environ


def test_env_file_search_order(tmp_path, monkeypatch):
    a = tmp_path / "a.env"
    a.write_text("MIMO_API_KEY=a\n")
    (tmp_path / ".env").write_text("MIMO_API_KEY=cwd\n")
    monkeypatch.setenv("ADAM_TTS_ENV_FILE", str(a))
    assert Secrets().get("MIMO_API_KEY") == "a"
    b = tmp_path / "b.env"
    b.write_text("MIMO_API_KEY=b\n")
    assert Secrets(str(b)).get("MIMO_API_KEY") == "b"  # --env-file beats ADAM_TTS_ENV_FILE


def test_home_config_env(tmp_path):
    d = tmp_path / "home" / ".config" / "adam_tts"
    d.mkdir(parents=True)
    (d / ".env").write_text("DOUBAO_API_KEY=h\n")
    assert Secrets().source("DOUBAO_API_KEY") == ".env"


def test_missing_and_redact(monkeypatch):
    s = Secrets()
    assert s.source("MIMO_API_KEY") == "missing" and s.get("MIMO_API_KEY") is None
    monkeypatch.setenv("MIMO_API_KEY", FAKE_KEY)
    s = Secrets()
    assert FAKE_KEY not in s.redact(f"error: bad key {FAKE_KEY}!")
    assert s.report() == {"MIMO_API_KEY": "env", "DOUBAO_API_KEY": "missing"}


def test_explicit_env_file_missing(tmp_path):
    assert Secrets(str(tmp_path / "nope.env")).missing_explicit
