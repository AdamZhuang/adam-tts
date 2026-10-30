# adam_tts 设计文档（v1）

> 状态：草案 · 作者：Adam Zhuang（AdamZhuang）· 日期：2026-10-30
> 仓库：`AdamZhuang/adam-tts`（公开，待建）· 包名 / CLI：`adam_tts` · Python ≥ 3.11
> 本文只做设计，不含实现代码。
> 已定：豆包只用新版控制台 API Key；缓存放在各输出目录内；MIT 协议；不发 PyPI，统一 `uv tool install/upgrade`；qwen/mimo 的 `--speed` 报错。

## 1. 目标与非目标

**目标**
- 一个给 agent 用的 TTS 工作台：CLI 为唯一入口，`skills/adam_tts/SKILL.md` 教 agent 怎么用。以后的 GUI 也只调 CLI。
- 每个命令都是**原子能力**：`say`、`synth`、`voice list`、`align`、`status`、`doctor`。
- 后端可插拔：`qwen`（本地）、`mimo`、`edge`、`doubao`。统一接口：文本 + 音色 + 风格（+ 可选语速）→ wav。
- 按**场景**整段合成，再用强制对齐切回逐行 wav，支持局部重跑和按内容哈希缓存。
- 所有命令都支持 `--json`，退出码有文档，方便 agent 判断。

**非目标（v1 不做）**
- `init` 命令（配置文件由用户手写，没有就用内置默认值）。
- 声音克隆、声音设计。
- 跨后端的音色别名：`--voice` 只认该后端的原生音色名。
- `render`（ffmpeg 出视频）、GUI：放到后续阶段。
- 自动降级到 doubao：任何情况下都不会。

## 2. 仓库结构

```
adam-tts/
├── pyproject.toml            # uv 管理；extras: [qwen] [align]；dev 依赖组
├── README.md  LICENSE(MIT)  .gitignore(.env, .cache/, out/)  .env.example
├── docs/design.md
├── src/adam_tts/
│   ├── __main__.py           # python -m adam_tts
│   ├── cli.py                # argparse 命令定义，只做参数解析 + 输出
│   ├── service.py            # 各命令的实现（与 argparse 解耦，便于测试）
│   ├── errors.py             # 异常类型 ↔ 退出码
│   ├── output.py             # 人类可读 / --json 两种输出
│   ├── config.py             # config.toml 读取、预设、参数合并
│   ├── secrets.py            # 环境变量 + .env 解析，脱敏
│   ├── tsv.py                # lines.tsv 读取与校验
│   ├── textsplit.py          # 超长场景按句切块
│   ├── audio.py              # wav 读写、重采样、拼接、ffmpeg 调用
│   ├── cache.py  manifest.py
│   ├── pipeline/synth.py     # synth 流水线
│   ├── align/
│   │   ├── base.py           # Aligner Protocol
│   │   ├── qwen_aligner.py   # Qwen3-ForcedAligner-0.6B
│   │   └── srt.py
│   └── backends/
│       ├── base.py           # TTSBackend Protocol、Capabilities、SynthRequest
│       ├── registry.py
│       ├── qwen.py  mimo.py  edge.py  doubao.py   # M1 中 qwen/doubao 为 stubs.py 里的占位
│       └── data/             # doubao_voices.json（静态音色表；mimo 的 9 个音色直接写在 mimo.py）
├── skills/adam_tts/SKILL.md   # Skill 唯一来源；本地 agent 的 skills 目录软链到这里
└── tests/
    ├── unit/                 # FakeBackend / FakeAligner，无网络
    ├── smoke/                # 每个后端一个，按环境变量开关
    └── fixtures/lines.tsv
```

依赖：核心 `python-dotenv`、`requests`、`numpy`、`soundfile`、`edge-tts`（兜底后端随核心安装）。`[qwen]` = `qwen-tts`、`torch`；`[align]` = `qwen-asr`。ffmpeg 为系统依赖（edge 输出 mp3 需要解码）。

CLI 框架选 **argparse**：零依赖；退出码和 `--json` 错误输出完全自己控制（用法错误天然是 2）；命令少且扁平，用不上 typer 的类型推导。

## 3. 命令

全局参数：`adam_tts --version`（`--json` 时输出 `{"version": "x.y.z"}`），供 agent 检查最低版本。

通用参数（`say`/`synth` 共用）：
`--preset NAME` · `--backend {qwen,mimo,edge,doubao}` · `--voice NAME`（原生名）· `--style TEXT` · `--prefix TEXT`（行首内联标签，如 `(平静 干练)`）· `--speed FLOAT`（倍率 0.5–2.0）· `--pitch INT`（Hz 偏移，M1 仅 edge；doubao 的映射在 M3 定）· `--volume FLOAT`（倍率 0.5–2.0）· `--lang`（qwen 用，默认 zh）· `--env-file PATH` · `--config PATH` · `--json` · `-v`。

参数合并规则：**命令行显式值 > 预设 > 后端默认**。
- 什么都不传 = `--preset draft`。
- 只传 `--backend X` 不传 `--preset`：不套用任何预设，只用后端默认值（避免把 mimo 的 style 带给 edge）。
- `--preset` 与 `--backend` 冲突（预设的后端不是 X）→ 退出码 2。

### 3.1 `say` —— 单句快速合成
```
adam_tts say "要念的文本" -o out.wav [通用参数]
adam_tts say --text-file t.txt -o out.wav --backend edge --voice zh-CN-YunxiNeural --speed 1.1
```
输出：`out.wav`（24 kHz 单声道 16-bit）。`--json` 返回 `{ok, out, preset, backend, voice, style, prefix, speed, pitch, volume, lang, duration_s, elapsed_s, chunks, sample_rate}`。`say` 不走缓存（缓存只在 `synth` 的输出目录里）。文本过长同样按句切块后拼接。

### 3.2 `synth` —— 按场景批量合成
```
adam_tts synth lines.tsv -o out/ep01 [通用参数]
adam_tts synth lines.tsv -o out/ep01 --scene OP4          # 只重跑一个场景
adam_tts synth lines.tsv -o out/ep01 --only o3,o7         # 只重跑包含这些行的场景，只覆盖这些行
adam_tts synth lines.tsv -o out/ep01 --force              # 忽略缓存
```
其他参数：`--scene-gap 0.5`（full.wav 里场景之间的静音秒数）、`--no-align`（只出场景音频）、`--allow-fallback`（见 §6）、`--dry-run`（只打印计划和切块，不调用后端）。

输出目录：
```
out/ep01/
├── full.wav  full.srt
├── segments/seg_01_OP1.wav  seg_01_OP1.srt ...
├── lines/o1.wav ...
├── manifest.json
└── .cache/                 # 按内容哈希存的场景音频与对齐结果（缓存只放在各输出目录内，删目录即清缓存）
```
说明：`--only` 的粒度仍是场景（为了语气连贯，整场景重合成），但只替换指定行的 `lines/*.wav` 和 manifest 条目；`full.wav` 总是用最新的场景音频重新拼接。

### 3.3 `voice list`
```
adam_tts voice list --backend edge --lang zh-CN --json
```
qwen：读模型支持的 speaker；edge：`edge_tts.list_voices()`（联网）；mimo、doubao：读包内静态表（doubao 按 resource id 过滤 1.0/2.0）。输出字段：`name, lang, gender, description, supports_style`。

### 3.4 `align` —— 时间戳 + 字幕
```
adam_tts align audio.wav --text-file t.txt -o out/align
adam_tts align seg_01_OP1.wav --tsv lines.tsv --scene OP1 -o out/align
```
输出 `align.json`（字级时间戳 + 行/句级区间）和 `.srt`。给 TSV 时每行一条字幕；给纯文本时按句切字幕（`--max-cue-chars 20`）。对齐文本会先去掉内联标签前缀。`synth` 内部调用同一实现。

### 3.5 `status`
```
adam_tts status out/ep01 [--tsv lines.tsv] --json
```
读 manifest，报告每行 `done / missing / failed / stale`。传 `--tsv` 时与当前文本比对哈希，文本改过的行标为 `stale`。返回值附一条建议的重跑命令（如 `--only o3,o7`）。

### 3.6 `doctor`
```
adam_tts doctor [--backend mimo] [--deep] --json
```
检查项：
- 密钥：每个变量报告 `set(env) / set(.env) / missing`，**绝不输出值或前后缀**。
- 后端可达：mimo/doubao 做 HTTPS 连通 + 鉴权轻探测；edge 拉音色表；qwen 检查 `qwen_tts` 可导入、模型目录存在、`torch.backends.mps.is_available()`。
- 对齐器：`qwen_asr` 可导入、`Qwen3-ForcedAligner-0.6B` 目录存在。
- ffmpeg：在 PATH 上且能运行。
- `--deep`：每个已配置后端真合成一句短文本（会产生费用；doubao 仍需 `--backend doubao` 显式指定才测）。

### 3.7 输出与退出码

`--json` 时 stdout 只有一个 JSON 对象，日志全部走 stderr。失败形如 `{"ok":false,"error":{"code":4,"type":"UnsupportedParam","message":"...","hint":"..."}}`。

| 码 | 含义 |
|---|---|
| 0 | 成功 |
| 1 | 未分类内部错误 |
| 2 | 用法错误（参数缺失/冲突、TSV 格式错误） |
| 3 | 配置或密钥缺失 |
| 4 | 后端不支持该参数（能力不匹配） |
| 5 | 后端调用失败（网络、HTTP、配额、鉴权） |
| 6 | 对齐失败 |
| 7 | 部分成功（synth 有行失败；manifest 已写） |
| 8 | 依赖缺失（ffmpeg、模型、可选 extra） |

`doctor` 有任一必需项失败时退出 3 或 8。

## 4. 后端接口与能力矩阵

```python
class Capabilities(TypedDict):
    style: bool; prefix: bool; speed: bool; pitch: bool; volume: bool
    max_text_chars: int | None; max_text_bytes: int | None
    sample_rate: int

@dataclass(frozen=True)
class SynthRequest:
    text: str; voice: str
    style: str | None = None; prefix: str | None = None
    speed: float | None = None; pitch: float | None = None; volume: float | None = None
    lang: str | None = None

class TTSBackend(Protocol):
    name: str
    version: str                      # 适配器版本，进入缓存键
    def capabilities(self, voice: str | None = None) -> Capabilities: ...
    def list_voices(self, lang: str | None = None) -> list[VoiceInfo]: ...
    def check(self, deep: bool = False) -> list[CheckResult]: ...
    def synthesize(self, req: SynthRequest, out_wav: Path) -> SynthResult: ...
```
公共层在调用 `synthesize` 前按 `capabilities()` 校验参数，后端不支持就抛 `UnsupportedParam`（退出码 4）；后端内部不再做静默忽略。输出统一重采样为 24 kHz 单声道 16-bit wav。

| 能力 | qwen（本地） | mimo | edge | doubao |
|---|---|---|---|---|
| 调用方式 | 进程内 `qwen-tts`，MPS fp16 | HTTPS chat.completions | `edge-tts` | HTTPS `/api/v3/tts/unidirectional`（chunked） |
| 音色 | speaker（Serena、Eric…） | 苏打、冰糖、茉莉、白桦… | `zh-CN-*Neural` | `speaker` id |
| `--style` | ✅ → `instruct` | ✅ → user 消息 | ❌ 报错 | ✅ → `context_texts[0]`（仅 2.0 音色，1.0 报错） |
| `--prefix` | ✅ 拼在文本前 | ✅ 拼在 assistant 文本前 | ❌ | ❌ |
| `--speed` | ❌ 报错 | ❌ 报错 | ✅ → `rate="+N%"` | ✅ → `speech_rate=(speed-1)*100`，范围 [-50,100] |
| `--pitch/--volume` | ❌ | ❌ | ✅ | ✅ `post_process.pitch` / `loudness_rate` |
| 单次文本上限 | 配置项（默认 600 字） | 配置项（默认 600 字，待确认） | 配置项（默认 2000 字） | 约 1024 字节（UTF-8 约 300 汉字，超限错误码 40402003） |
| 自动选用 | 否 | 是（draft） | 是（兜底） | **否，只能 `--backend doubao`** |

后端要点：
- **qwen**：在 Mac mini 上进程内加载 Qwen3-TTS CustomVoice 1.7B（路径来自配置，默认 `~/qwen3-tts`），模型常驻一次进程内复用。`pip install adam_tts[qwen]`；未安装时报退出码 8。参考现有 `scripts/batch_custom_voice.py`、`batch_scene_voice.py` 的调用方式。
- **mimo**：`POST https://api.xiaomimimo.com/v1/chat/completions`，`model=mimo-v2.5-tts`；消息 = `user`(style，不朗读) + `assistant`(prefix+text)；`audio={format:wav, voice}`；读 `choices[0].message.audio.data` base64。参照 `/workspace/mimo/mimo_tts.py`。
- **edge**：输出 mp3，经 ffmpeg 转 wav。
- **doubao**：音频格式请求 `pcm`（官方说明流式下 wav 会重复返回头），本地封装 wav；`X-Api-Resource-Id` 默认 `seed-tts-2.0`；记录响应头 `X-Tt-Logid` 到 manifest 便于排查。

**`--speed` 决策：qwen / mimo 一律报错，不翻译成风格指令。** 理由：(1) 规则统一——不支持就报错，agent 看到退出码 4 就知道该怎么办；(2) 翻译出来的"语速稍快"无法保证效果，数值和实际语速对不上，反而误导；(3) 语速本来就可以写进 `--style`（draft 预设已含"语速中等偏快"），报错提示里直接给出这条建议。

## 5. 配置、预设与密钥

**配置文件** `~/.config/adam_tts/config.toml`（可用 `--config` 或 `ADAM_TTS_CONFIG` 覆盖；不存在就用内置默认值，内置值与下例相同）：
```toml
default_preset  = "draft"
fallback_preset = "edge"

[presets.draft]            # 也叫 calm2：平静解说
backend = "mimo"
voice   = "苏打"
style   = "平铺直叙的解说旁白，语速中等偏快，语调平稳，起伏很小，不强调重音，不拖长音，不要抑扬顿挫，不要播音腔，像工程师在平静地讲解原理。"
prefix  = "(平静 干练)"

[presets.edge]
backend = "edge"
voice   = "zh-CN-YunxiNeural"

[backends.qwen]
home = "~/qwen3-tts"
model = "models/Qwen3-TTS-12Hz-1.7B-CustomVoice"   # 以实际目录为准
device = "mps"; dtype = "float16"

[aligner]
model = "~/qwen3-tts/models/Qwen3-ForcedAligner-0.6B"

[backends.doubao]
resource_id = "seed-tts-2.0"
```
校验：`default_preset` / `fallback_preset` 不允许指向 doubao；v1 预设里也不允许 `backend = "doubao"`，doubao 只能经 `--backend doubao` 使用。

**密钥解析顺序**（只读不写）：
1. 进程环境变量（最高优先）。
2. `.env` 文件补充，`python-dotenv` 且 `override=False`。只加载第一个找到的：`--env-file` > `$ADAM_TTS_ENV_FILE` > `./.env` > `~/.config/adam_tts/.env`。

`.env.example`（只有变量名）：
```
MIMO_API_KEY=
# 豆包（火山引擎，新版控制台）
DOUBAO_API_KEY=
# 可选，默认 seed-tts-2.0
DOUBAO_RESOURCE_ID=
# 可选：本地 qwen 路径（非密钥，也可写在 config.toml）
ADAM_TTS_QWEN_HOME=
```
豆包只用新版控制台鉴权（已对照火山引擎官方文档 HTTP Chunked/SSE 单向流式 V3，docs/6561/1598757）：请求头 `X-Api-Key`（= `DOUBAO_API_KEY`）+ `X-Api-Resource-Id`（= `DOUBAO_RESOURCE_ID`，默认 `seed-tts-2.0`）。不支持旧版 APP ID / Access Token。qwen 走本地，不需要远程密钥。

**脱敏**：密钥值不进日志、manifest、异常信息和 `--json` 输出；HTTP 错误体回显前做脱敏替换；单测会扫描所有输出，确认假密钥不出现。

## 6. synth 流水线与 manifest

1. **读 TSV**：列 `id, scene, text`，UTF-8，允许表头行；校验 id 唯一、同一场景的行连续。按 `--scene/--only` 选出要跑的场景。
2. **解析参数**：按 §3 合并，做能力校验（不通过直接退出 4，不发任何请求）。
3. **切块**：场景文本 = 各行直接拼接。超过后端上限时，优先在行边界切，单行仍超限再在句末标点（。！？；…）切，绝不在句中切。每块都加 `prefix`、同一个 `style`。
4. **缓存键**：`sha256(backend, adapter_version, model, voice, style, prefix, speed, pitch, volume, lang, chunk_text)`。命中则复用 `.cache/` 中的音频，否则调用后端。
5. **合成**：逐块合成，块之间按顺序拼接为场景音频 `segments/seg_XX_<scene>.wav`（XX 为场景在 TSV 中的序号）。
6. **对齐切行**：对每块音频做强制对齐（文本不含 prefix），得到字级时间戳，偏移后合并为场景时间轴。每行区间 = 首字开始到末字结束；相邻两行的切点取两者之间静音的中点；首尾各留 50 ms。写 `lines/{id}.wav` 和场景 srt。
7. **拼 full**：按 TSV 顺序拼接所有场景，场景间插 `--scene-gap` 静音，写 `full.wav` / `full.srt`。
8. **写 manifest**：每完成一个场景就原子写入一次（临时文件 + rename），中途失败也能 `status`。

**跳过与重跑**：某行 manifest 状态为 `done`、`text_hash` 与当前一致、文件存在 → 跳过。场景内只要有一行要重跑，整场景重合成（命中缓存则不花钱）。

**降级**：默认不降级——一集里混用不同声音比失败更糟。传 `--allow-fallback` 时，后端调用失败（退出码 5 类）的场景改用 `fallback_preset`（edge）重跑，manifest 标记 `fallback: true`。永远不会降级到 doubao。

**manifest.json（schema_version 1）**
```json
{
  "schema_version": 1,
  "tool_version": "0.1.0",
  "created_at": "2026-10-30T22:00:00+08:00",
  "input": {"tsv": "lines.tsv", "tsv_sha256": "..."},
  "params": {"preset": "draft", "backend": "mimo", "model": "mimo-v2.5-tts",
             "voice": "苏打", "style": "...", "prefix": "(平静 干练)", "speed": null},
  "sample_rate": 24000,
  "full": {"path": "full.wav", "srt": "full.srt", "duration_s": 312.4},
  "scenes": [{
    "scene": "OP1", "index": 1, "path": "segments/seg_01_OP1.wav",
    "offset_s": 0.0, "duration_s": 18.2, "status": "done",
    "chunks": [{"cache_key": "...", "chars": 96, "cache_hit": false,
                "elapsed_s": 6.1, "request_id": null}],
    "backend": "mimo", "fallback": false, "error": null
  }],
  "lines": [{
    "id": "o1", "scene": "OP1", "text_hash": "...", "status": "done",
    "path": "lines/o1.wav", "start_s": 0.05, "end_s": 4.10,
    "align_confidence": 0.93, "error": null
  }]
}
```
行状态：`done / failed / missing`（`status` 额外推导 `stale`）。

## 7. 错误处理

- 异常类型一一对应退出码（§3.7），CLI 顶层统一捕获并按 `--json` 输出；带 `hint` 告诉 agent 下一步（例如“去掉 --speed，把语速写进 --style”）。
- 网络错误：指数退避重试 3 次（只对超时、5xx、429）；4xx 鉴权错误不重试，直接退出 5 并提示跑 `doctor`。
- 对齐失败或置信度低于阈值：该场景音频保留，行标 `failed`，整体退出 7；不会丢已合成的音频。
- 输出完整性：所有文件先写临时文件再 rename，避免半个 wav。
- 输入校验早失败：TSV 错误、音色不在已知表（mimo/doubao 静态表、edge 在线表）在发请求前报 2。

## 8. 测试策略

- **单元测试（默认跑，无网络、无密钥）**：
  - `FakeBackend`：按文本长度生成定长正弦/静音 wav，可配置能力和失败；`FakeAligner`：按字均分时间戳。
  - 覆盖：参数合并与冲突、能力校验（每个 ❌ 格都有用例）、TSV 解析、切块边界、缓存键稳定性、manifest 读写与原子性、`status` 推导、`--only/--scene` 选择、降级逻辑与“不降级到 doubao”、`--json` 结构与退出码、密钥解析顺序（env 优先、`.env` 不覆盖）、输出中不含假密钥。
  - 后端适配器用录制的 HTTP 响应（mimo base64、doubao chunked 流）做请求构造与解析测试。
- **冒烟测试（`pytest -m smoke`，按环境开关）**：每个后端一个，合成一句短文本并检查 wav 可读、时长 > 0。
  - mimo：需 `ADAM_TTS_SMOKE=1` 且 `MIMO_API_KEY`；edge：`ADAM_TTS_SMOKE=1`；qwen / 对齐器：模型目录存在才跑；doubao：还需 `ADAM_TTS_SMOKE_DOUBAO=1`（产生费用，必须显式开）。
- CI（GitHub Actions）只跑单元测试和 lint，不配任何密钥。

## 9. 部署与调用方式

**安装位置**：CLI 只装在 Mac mini（Apple M4）上，统一用 uv 从 GitHub 安装（不发布到 PyPI）：`uv tool install git+https://github.com/AdamZhuang/adam-tts`（需要本地 qwen/对齐时用 `uv tool install "adam_tts[qwen,align] @ git+https://github.com/AdamZhuang/adam-tts"`），得到全局命令 `adam_tts`。Qwen 模型、对齐器模型、`.env` 密钥、输出目录全部在这台机器上。其他机器不安装。

**接口契约**：agent 与工具之间**唯一**的契约是 CLI 参数 + `--json` 的 stdout + 退出码。agent 不 import 包，也不读内部文件（manifest 通过 `status --json` 读）。

**Skill**：唯一来源是仓库里的 `skills/adam_tts/SKILL.md`，只写怎么调用（何时用、标准流程、参数、配音策略、退出码处理），不写实现。CLI 内部改动不需要改 Skill，只有接口变化才改。

**两类调用方**
- **本地 agent**（Mac mini 上的 Qoder、DeepSeek Harness、Claude Code 等）：直接执行 `adam_tts ...`。各自的 skills 目录软链到仓库的 `skills/adam_tts/`，`git pull` 一次全部同步。
- **远程 agent**（跑在各自云主机上的 Grok Bot）：通过远程 shell 在 Mac mini 上执行同样的命令，每次需用户批准。文件交换走约定目录 `~/adam_tts_jobs/<job>/`：
  ```
  ~/adam_tts_jobs/<job>/
  ├── lines.tsv          # 远程 agent 拷入
  └── out/               # adam_tts synth lines.tsv -o out 的输出
  ```
  流程：拷入 `lines.tsv` → 远程执行 `cd ~/adam_tts_jobs/<job> && adam_tts synth lines.tsv -o out --json` → 按退出码处理 → 把 `out/`（或只拷 `full.wav`、`lines/`、`manifest.json`）拷回自己的机器。`<job>` 用 `<日期>-<剧集或任务名>`，如 `20261030-ep01`。

**版本与发布**
- `adam_tts --version` 报版本；SKILL.md 顶部写明最低 CLI 版本（如 `min_cli_version: 0.2.0`）。agent 先查版本，低于要求就提示用户升级，不自行继续。
- 发布流程：改 CLI → 接口有变就同步改 SKILL.md（并提高最低版本）→ 打 git tag → 在 Mac mini 上执行 `uv tool upgrade adam_tts`。

**以后（非 v1）**：在 Mac mini 上加一个薄的 HTTP 或 MCP 封装，内部仍调用同一个 CLI，让远程调用不再依赖远程 shell 和手动拷文件。

## 10. SKILL.md 大纲（skills/adam_tts/SKILL.md）

1. **何时使用**：需要配音、口播草稿、把台本转成逐行 wav、给音频出字幕/时间戳时。
2. **前置检查**：`adam_tts --version` 不低于 SKILL.md 写明的最低版本，否则提示用户升级；再跑 `adam_tts doctor --json`；缺密钥就告诉用户变量名，不要索要或打印值。
3. **标准流程**：准备 `lines.tsv`（id/scene/text）→ `adam_tts synth lines.tsv -o out/<ep> --json` → 看退出码 → `adam_tts status out/<ep> --json` → 报告 `full.wav` 与 `lines/` 路径。单句试听用 `say`。
4. **局部重跑**：改了文本或某几行不满意 → `status --tsv` 找 stale/failed → `synth ... --only id1,id2` 或 `--scene OP4`；不要整集重跑。
5. **配音策略**：
   - 草稿默认 `draft` 预设：mimo · 苏打 · calm2 风格 + `(平静 干练)`。
   - mimo 不可用时用 `--preset edge`，并告诉用户换了声音。
   - doubao 只在用户明确要求时用 `--backend doubao`，绝不自动选。
   - 旁白要平静、不夸张、不播音腔；不要自行加情绪化的 style。
6. **参数规则**：`--voice` 用后端原生名（先 `voice list`）；不支持的参数会报退出码 4，按 hint 改，不要换后端绕过。
7. **退出码速查**与常见错误处理。
8. **本地调用**：Mac mini 上的 agent 直接运行命令；skills 目录是指向仓库的软链，不要复制。
9. **远程调用**：只在用户批准后经远程 shell 在 Mac mini 上运行；用 `~/adam_tts_jobs/<job>/` 放 `lines.tsv` 和 `out/`；完成后把结果拷回本机；不要在远程机器上安装 adam_tts，也不要读取或拷走 Mac mini 上的 `.env`。

## 11. 里程碑

- **M1**：CLI 骨架（argparse、`--json`、退出码、配置/预设/密钥）+ mimo + edge 后端 + `say`、`voice list`、`doctor`、`--version` + 单元测试与 CI；在 Mac mini 上用 uv 装好。
- **M2**：`synth` 场景级流水线 + 切块 + 缓存 + manifest + `align`（Qwen3-ForcedAligner）+ `status` + qwen 后端。
- **M3**：doubao 后端 + 完整 SKILL.md（含最低版本、本地/远程调用）+ 本地 agent 软链 + 用一集真实台本跑通（mimo 草稿 + 局部重跑），本地和远程（`~/adam_tts_jobs/`）各跑一次。
- **后续**：`render`（ffmpeg 出视频）、GUI（调用 CLI）、HTTP/MCP 薄封装、声音克隆、声音设计。

## 12. 待定问题

1. MiMo 单次请求的文本上限、返回 wav 的采样率、限流与计费——需实测后定默认切块长度。
2. MiMo 有无音色列表 API？没有就维护静态表。
3. 豆包：显式使用时默认哪个 2.0 音色？1024 字节上限是否也适用于 V3 HTTP 接口（文档计费页写的是按次 1024 字节）？
4. qwen-tts 与 qwen-asr 能否装进 uv 为 adam_tts 建的同一个隔离环境（目前对齐器在单独的 `~/qwen3-tts/aligner/.venv`）？不能的话，对齐器改为调用配置里指定的 python 子进程（JSON 走 stdio）。
5. ForcedAligner 单次能处理的最长音频？长场景是否需要按块对齐（当前设计是按块对齐）。
6. 默认 `--scene-gap`（暂定 0.5 s）和行切点策略（静音中点 + 50 ms 余量）是否合适。
7. 可选想法（不进 v1）：加 `adam_tts skill install --target <dir>`，自动为本地 agent 的 skills 目录建软链。
8. `~/adam_tts_jobs/` 下旧任务由谁清理、保留多久。
