# adam_tts

给 agent 用的 TTS 工作台：一个 CLI（`adam_tts`）+ 一份 Skill（`skills/adam_tts/SKILL.md`，M3）。
后端可插拔：`mimo`（小米 MiMo TTS，默认草稿）、`edge`（Edge TTS，兜底）、`qwen`（本地 Qwen3-TTS，M2）、`doubao`（豆包语音，M3，仅显式指定时使用）。

设计文档：[docs/design.md](docs/design.md)

## 安装（Mac mini）

```bash
uv tool install git+https://github.com/AdamZhuang/adam-tts
uv tool upgrade adam_tts      # 升级
```

系统依赖：ffmpeg（`brew install ffmpeg`）。

## 密钥

先读环境变量，再用 `.env` 补充（不覆盖已有环境变量）。查找顺序：`--env-file` > `$ADAM_TTS_ENV_FILE` > `./.env` > `~/.config/adam_tts/.env`。变量名见 [.env.example](.env.example)。工具不会输出任何密钥值。

## 用法（M1）

```bash
adam_tts --version
adam_tts doctor --json                       # 检查密钥（只显示来源）、后端、ffmpeg
adam_tts voice list --backend mimo
adam_tts say "要念的文本" -o out.wav          # 默认 draft 预设：mimo · 苏打 · 平静解说
adam_tts say "要念的文本" -o out.wav --preset edge
adam_tts say "要念的文本" -o out.wav --backend edge --voice zh-CN-YunxiNeural --speed 1.1
```

每个命令都支持 `--json`（stdout 只有一个 JSON 对象）。退出码：0 成功 · 2 用法错误 · 3 配置/密钥缺失 · 4 后端不支持该参数 · 5 后端调用失败 · 6 对齐失败 · 7 部分成功 · 8 依赖缺失/后端未实现。

## 开发

```bash
uv sync
uv run pytest                                # 单元测试（无网络）
ADAM_TTS_SMOKE=1 uv run pytest -m smoke      # 真实调用 mimo / edge
```

## 协议

MIT
