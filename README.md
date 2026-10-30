# adam_tts

给 agent 用的 TTS 工作台：一个 CLI（`adam_tts`）+ 一份 Skill（`skills/adam_tts/SKILL.md`，M3）。
后端可插拔：`mimo`（小米 MiMo TTS，默认草稿）、`edge`（Edge TTS，兜底）、`qwen`（本地 Qwen3-TTS，M2）、`doubao`（豆包语音，M3，仅显式指定时使用）。

设计文档：[docs/design.md](docs/design.md)

## 安装（Mac mini）

```bash
uv tool install git+https://github.com/AdamZhuang/adam-tts
uv tool upgrade adam_tts      # 升级
```

不需要 ffmpeg（音频读写用 soundfile；装了 ffmpeg 会作为其它格式的兜底解码器）。

本地 qwen 后端和对齐器默认复用 `~/qwen3-tts` 里已有的两个环境（`.venv`、`aligner/.venv`），在子进程里加载模型，所以 adam_tts 本身不需要装 torch。路径用 `ADAM_TTS_QWEN_HOME` 改。

## 密钥

先读环境变量，再用 `.env` 补充（不覆盖已有环境变量）。查找顺序：`--env-file` > `$ADAM_TTS_ENV_FILE` > `./.env` > `~/.config/adam_tts/.env`。变量名见 [.env.example](.env.example)。工具不会输出任何密钥值。

## 用法

```bash
adam_tts --version
adam_tts doctor --json                       # 检查密钥（只显示来源）、后端、ffmpeg
adam_tts voice list --backend mimo
adam_tts say "要念的文本" -o out.wav          # 默认 draft 预设：mimo · 苏打 · 平静解说
adam_tts say "要念的文本" -o out.wav --preset edge
adam_tts say "要念的文本" -o out.wav --backend edge --voice zh-CN-YunxiNeural --speed 1.1

# 批量：lines.tsv = id<TAB>scene<TAB>text
adam_tts synth lines.tsv -o out/ep01 --backend qwen --voice Serena --style "语气平静克制"
adam_tts synth lines.tsv -o out/ep01 --scene OP1,OP2      # 只做这些场景
adam_tts synth lines.tsv -o out/ep01 --only o3,o7         # 改了几行后只更新它们
adam_tts status out/ep01 --tsv lines.tsv                  # done/missing/failed/stale + 建议的重跑命令
adam_tts align seg.wav --tsv lines.tsv --scene OP1 -o out/align   # 时间戳 JSON + SRT
```

`synth` 输出：`full.wav`/`full.srt`、`segments/seg_XX_<scene>.wav`/`.srt`、`lines/<id>.wav`、`manifest.json`、`.cache/`（内容哈希缓存，重跑不重复花钱）。

每个命令都支持 `--json`（stdout 只有一个 JSON 对象）。退出码：0 成功 · 2 用法错误 · 3 配置/密钥缺失 · 4 后端不支持该参数 · 5 后端调用失败 · 6 对齐失败 · 7 部分成功 · 8 依赖缺失/后端未实现。

## 开发

```bash
uv sync
uv run pytest                                # 单元测试（无网络）
ADAM_TTS_SMOKE=1 uv run pytest -m smoke      # 真实调用 mimo / edge
```

## 协议

MIT
