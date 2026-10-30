"""User config (~/.config/adam_tts/config.toml), presets and parameter merging.

Merge order: explicit CLI value > preset > backend default.
"""

from __future__ import annotations

import copy
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from adam_tts.errors import ConfigError, UsageError

CALM2_STYLE = (
    "平铺直叙的解说旁白，语速中等偏快，语调平稳，起伏很小，不强调重音，不拖长音，"
    "不要抑扬顿挫，不要播音腔，像工程师在平静地讲解原理。"
)

BUILTIN_CONFIG: dict = {
    "default_preset": "draft",
    "fallback_preset": "edge",
    "presets": {
        "draft": {  # a.k.a. calm2
            "backend": "mimo",
            "voice": "苏打",
            "style": CALM2_STYLE,
            "prefix": "(平静 干练)",
        },
        "edge": {
            "backend": "edge",
            "voice": "zh-CN-YunxiNeural",
        },
    },
    "backends": {
        "mimo": {"base_url": "https://api.xiaomimimo.com/v1", "model": "mimo-v2.5-tts",
                 "max_text_chars": 600, "timeout_s": 300},
        "edge": {"max_text_chars": 2000},
        "qwen": {"home": "~/qwen3-tts"},
        "doubao": {"resource_id": "seed-tts-2.0"},
    },
}

PARAM_FIELDS = ("voice", "style", "prefix", "speed", "pitch", "volume", "lang")
# backends that may never be selected implicitly (default/fallback/preset)
EXPLICIT_ONLY_BACKENDS = {"doubao"}


def default_config_path() -> Path:
    if os.environ.get("ADAM_TTS_CONFIG"):
        return Path(os.environ["ADAM_TTS_CONFIG"]).expanduser()
    return Path("~/.config/adam_tts/config.toml").expanduser()


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Config:
    data: dict
    path: Path | None = None  # file actually loaded, if any

    @property
    def presets(self) -> dict:
        return self.data.get("presets", {})

    def backend_settings(self, name: str) -> dict:
        return dict(self.data.get("backends", {}).get(name, {}))

    def preset(self, name: str) -> dict:
        if name not in self.presets:
            raise UsageError(f"未知预设: {name}", hint=f"可用预设: {', '.join(sorted(self.presets))}")
        return dict(self.presets[name])


def validate(data: dict, known_backends: set[str]) -> None:
    presets = data.get("presets", {})
    for key in ("default_preset", "fallback_preset"):
        name = data.get(key)
        if name and name not in presets:
            raise ConfigError(f"{key} 指向不存在的预设: {name}")
    for name, p in presets.items():
        b = p.get("backend")
        if not b:
            raise ConfigError(f"预设 {name} 缺少 backend")
        if b not in known_backends:
            raise ConfigError(f"预设 {name} 的 backend 未知: {b}")
        if b in EXPLICIT_ONLY_BACKENDS:
            raise ConfigError(f"预设 {name} 不能使用 {b}", hint=f"{b} 只能通过 --backend {b} 显式使用")


def load_config(path: str | None, known_backends: set[str]) -> Config:
    p = Path(path).expanduser() if path else default_config_path()
    data = copy.deepcopy(BUILTIN_CONFIG)
    loaded = None
    if p.is_file():
        try:
            with open(p, "rb") as f:
                user = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"配置文件解析失败: {p}: {e}") from e
        data = _deep_merge(data, user)
        loaded = p
    elif path:
        raise ConfigError(f"配置文件不存在: {p}")
    validate(data, known_backends)
    return Config(data=data, path=loaded)


@dataclass
class ResolvedParams:
    backend: str
    preset: str | None
    voice: str | None = None
    style: str | None = None
    prefix: str | None = None
    speed: float | None = None
    pitch: int | None = None
    volume: float | None = None
    lang: str | None = None
    explicit: set[str] = field(default_factory=set)  # fields given on the command line

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("preset", "backend", *PARAM_FIELDS)}


def resolve_params(cfg: Config, *, preset: str | None, backend: str | None, cli: dict) -> ResolvedParams:
    """cli: field -> value or None (None = not given). Empty string clears a preset value."""
    if preset is None and backend is None:
        preset = cfg.data.get("default_preset")
    base: dict = {}
    if preset is not None:
        base = cfg.preset(preset)
        pb = base.pop("backend")
        if backend is not None and backend != pb:
            raise UsageError(
                f"--preset {preset} 使用后端 {pb}，与 --backend {backend} 冲突",
                hint="只传其一：--preset 选一整套参数，或只用 --backend 走后端默认值",
            )
        backend = pb
    explicit = set()
    merged = {}
    for f in PARAM_FIELDS:
        v = cli.get(f)
        if v is not None:
            explicit.add(f)
            merged[f] = None if v == "" else v
        else:
            merged[f] = base.get(f)
    return ResolvedParams(backend=backend, preset=preset, explicit=explicit, **merged)
