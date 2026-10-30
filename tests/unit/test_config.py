import pytest

from adam_tts import config
from adam_tts.errors import ConfigError, UsageError

KNOWN = {"qwen", "mimo", "edge", "doubao", "fake"}


def cfg():
    return config.load_config(None, KNOWN)


def test_builtin_draft_preset():
    c = cfg()
    assert c.data["default_preset"] == "draft"
    d = c.preset("draft")
    assert d["backend"] == "mimo" and d["voice"] == "苏打"
    assert d["prefix"] == "(平静 干练)"
    assert d["style"].startswith("平铺直叙的解说旁白")


def test_default_is_draft_when_nothing_given():
    rp = config.resolve_params(cfg(), preset=None, backend=None, cli={})
    assert (rp.preset, rp.backend, rp.voice) == ("draft", "mimo", "苏打")


def test_backend_only_means_no_preset():
    rp = config.resolve_params(cfg(), preset=None, backend="edge", cli={})
    assert rp.preset is None and rp.style is None and rp.voice is None


def test_cli_overrides_preset_and_empty_clears():
    rp = config.resolve_params(cfg(), preset="draft", backend=None, cli={"voice": "白桦", "prefix": ""})
    assert rp.voice == "白桦" and rp.prefix is None and rp.style
    assert rp.explicit == {"voice", "prefix"}


def test_preset_backend_conflict():
    with pytest.raises(UsageError):
        config.resolve_params(cfg(), preset="draft", backend="edge", cli={})


def test_unknown_preset():
    with pytest.raises(UsageError):
        config.resolve_params(cfg(), preset="nope", backend=None, cli={})


def test_user_config_merges(fake_config):
    c = cfg()
    assert c.path == fake_config
    assert "draft" in c.presets and "fk" in c.presets
    assert c.data["default_preset"] == "fk"


@pytest.mark.parametrize("body", [
    '[presets.d]\nbackend = "doubao"\nvoice = "x"\n',
    'default_preset = "missing"\n',
    '[presets.d]\nbackend = "nosuch"\n',
])
def test_invalid_configs(tmp_path, monkeypatch, body):
    p = tmp_path / "bad.toml"
    p.write_text(body, encoding="utf-8")
    monkeypatch.setenv("ADAM_TTS_CONFIG", str(p))
    with pytest.raises(ConfigError):
        cfg()


def test_explicit_missing_config_path(tmp_path):
    with pytest.raises(ConfigError):
        config.load_config(str(tmp_path / "missing.toml"), KNOWN)
