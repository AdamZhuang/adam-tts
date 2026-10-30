---
name: adam_tts
description: 用 adam_tts CLI 做中文视频口播/配音合成：按 lines.tsv 按场景整段合成、再用强制对齐切回每句（lines/{id}.wav），并输出字幕（full.srt / 场景 srt）。用户要“配音”“口播”“念稿”“出音频/字幕”“重录某几句”时使用。
---

# adam_tts（适用 v0.2.x）

一个给 agent 用的 TTS 命令行。**整段（场景级）合成 → 对齐切句 → 字幕**，带缓存和 manifest，可局部重跑。

## 什么时候用
- 视频口播/解说配音：有一份逐句台本，要整集音频 + 每句单独音频 + 字幕。
- 改了几句台词，只重录这几句（其余不动、不重复花钱）。
- 已有一段音频 + 文本，要时间戳/SRT（`align`）。
- 单句试听：`adam_tts say "文本" -o x.wav`。

## 版本检查
```bash
adam_tts --version        # 需要 >= 0.2.1（synth/status/align/qwen 从 0.2.0 起有）
uv tool upgrade adam_tts  # 版本不够时升级
```
v0.2.x 可用后端：`mimo`（默认草稿）、`edge`（兜底）、`qwen`（本地 Qwen3-TTS）。**`doubao` 还没实现**（M3），用了会退出 8。

## 输入格式：lines.tsv
UTF-8，Tab 分隔三列 `id<TAB>scene<TAB>text`，表头行可有可无；`#` 开头的行忽略。
- `id` 全局唯一（用作 `lines/{id}.wav` 文件名）；同一 `scene` 的行必须连续；`text` 不能为空。
- 一个 scene 会被**整段一起合成**（语气连贯），所以按“一口气念完的一段”划分 scene，一般 2–10 句。
```
id	scene	text
o1	OP1	很多人用 Agent 写代码，容易陷入「传话困境」：
o2	OP1	它改完一版，你去跑、去看合不合格。
s1	S1A	先说结论：让 Agent 自己验收。
```

## 标准流程
```bash
J=~/adam_tts_jobs/<job>          # 每个任务一个目录
adam_tts doctor --json            # 1. 体检（密钥只报来源，不显示值）
# 2a. 草稿（默认 draft 预设 = MiMo · 苏打 · calm2 平静解说；需要 MIMO_API_KEY）
adam_tts synth $J/lines.tsv -o $J/out_mimo --json
# 2b. 本地 qwen · Serena（不花钱，约 2 倍实时）
adam_tts synth $J/lines.tsv -o $J/out_qwen --backend qwen --voice Serena \
  --style "语气平静克制，像技术讲解，不要夸张不要表演感；语速比正常讲解略快一点，吐字清楚，停顿短而干净" --json
adam_tts status $J/out_qwen --tsv $J/lines.tsv --json      # 3. 核对：应 complete
# 4.（可选）对任意音频单独出时间戳/字幕
adam_tts align $J/out_qwen/segments/seg_01_OP1.wav --tsv $J/lines.tsv --scene OP1 -o $J/align --json
adam_tts align some.wav --text-file t.txt -o $J/align --json  # 纯文本按句切字幕（--max-cue-chars 20）
```
- 只给 `--backend` 时**不套用预设**（不会带上 mimo 的 style/prefix），其余参数自己给全。
- 先 `--dry-run` 可看切块与缓存命中计划，不调用后端。
- `synth` 结果 JSON 里看 `lines.done/total_selected`、`weak_cuts`（切点没落在静音里的行，建议人工听一下）、`full_duration_s`。

## 局部重跑
```bash
adam_tts synth lines.tsv -o OUT --only o3,o7    # 改了这几行：重做所在场景，只替换这几行的 wav
adam_tts synth lines.tsv -o OUT --scene OP4     # 重做某些场景（逗号分隔）
adam_tts synth lines.tsv -o OUT                 # 直接全量重跑也行：没变的场景自动跳过（内容哈希缓存）
adam_tts synth lines.tsv -o OUT --scene OP4 --force   # 不满意同一文本的效果，强制重新合成
```
参数要和原来一致（同一后端/音色/style），否则会被视为新参数整场景重做。`status` 会给出建议的重跑命令（`suggested_command`）。`full.wav` 每次都会用最新场景音频重新拼。

## 输出目录
```
OUT/
├── full.wav  full.srt                 # 整集：场景间 0.5 s 静音（--scene-gap）
├── segments/seg_XX_<scene>.wav/.srt   # 每个场景的整段音频（XX = 场景在 TSV 中的序号）
├── lines/{id}.wav                     # 每句一个，首尾相接，切在句间静音处
├── manifest.json                      # 每行状态、时间、切点质量、参数；status 读它
└── .cache/                            # 内容哈希缓存，删掉只会导致重新合成
```
音频统一 24 kHz 单声道 16-bit wav。字幕时间用每句实际发声区间。

## 参数与后端能力
| 参数 | mimo | qwen | edge | doubao |
|---|---|---|---|---|
| `--voice` | 苏打/白桦（男）冰糖/茉莉（女）Mia/Chloe/Milo/Dean（英） | Serena, Vivian, Uncle_Fu, Ryan, Aiden, Ono_Anna, Sohee, Eric（川）, Dylan（京） | 如 zh-CN-YunxiNeural（`voice list --backend edge`） | 未实现 |
| `--style`（自然语言风格指令） | ✅ | ✅（作为 instruct） | ❌ | 未实现 |
| `--prefix`（行首标签，如 `(平静 干练)`） | ✅ | ✅ | ❌ | 未实现 |
| `--speed`（0.5–2.0 倍率） | ❌ 写进 style | ❌ 写进 style | ✅ | 未实现 |
| `--pitch`（Hz）/ `--volume`（0.5–2.0） | ❌ | ❌ | ✅ | 未实现 |
| `--lang` | ❌ | ✅（默认中文） | ❌ | 未实现 |
| 单次请求上限 | 600 字 | 400 字 | 2000 字 | — |
| 费用/位置 | 按量计费，需 MIMO_API_KEY | 本地 Mac mini，免费 | 免费，联网 | — |

不支持的参数**直接报错退出 4，不会静默忽略**。超长场景会在行边界（必要时句末）自动切块。`adam_tts voice list --backend <b> --json` 查音色。

## --json 与退出码
所有命令都支持 `--json`：stdout 只有**一个** JSON 对象（`ok` 字段 + 结果；失败时含 `error.code/type/message/hint`）；进度日志在 stderr。agent 应解析 stdout，按退出码处理：

| 码 | 含义 | agent 怎么做 |
|---|---|---|
| 0 | 成功 | 继续；看 `weak_cuts` |
| 2 | 用法错误（参数、TSV 格式、id 重复、场景不连续、未知音色） | 按 `message` 修命令或 TSV，再跑 |
| 3 | 配置/密钥缺失 | 告诉用户缺哪个变量（不要索要或打印值）；或改用 qwen |
| 4 | 后端不支持该参数 | 按 `hint` 去掉参数（如把语速写进 `--style`） |
| 5 | 后端调用失败（网络/鉴权/服务端） | 跑 `doctor`；稍后重试同一命令（缓存保留已完成部分）；是否 `--allow-fallback` 先问用户 |
| 6 | 对齐失败 | 场景音频已保留；`--scene X --force` 重做该场景，或 `--no-align` 只要整段 |
| 7 | 部分成功 | 看 `scenes.failed` / `status`，按 `suggested_command` 重跑缺的部分 |
| 8 | 依赖缺失 / 后端未实现（含 doubao） | 跑 `doctor` 看缺什么（模型目录、venv）；doubao 就换后端并告知用户 |
| 1 | 内部错误 | 带上 stderr 报告给用户 |

## 配音规则
- **草稿/测试默认 MiMo 苏打 calm2**（即不带参数的 `draft` 预设）。Mac mini 上没有 MIMO_API_KEY 时改用 qwen Serena + 上面的平静 instruct，并告知用户。
- **Edge 只做兜底**，且必须显式加 `--allow-fallback` 才会在后端失败时用；默认宁可失败也不混音色。
- **正式版用更好的声音（豆包）——但只在用户明确指定时用**；v0.2.x 尚未实现豆包，用户要正式版时说明这一点。
- 解说要**平静克制**：不加情绪化 style，不要播音腔/表演感。
- **整段合成，不要逐句 `say` 拼接**（逐句会语气断裂、停顿不一）；单句只用于试听。

## 本地调用（Mac mini）
CLI 装在 Mac mini（adamdeMac-mini.local）的 `~/.local/bin/adam_tts`（`uv tool install git+https://github.com/AdamZhuang/adam-tts`）。非交互 shell 不读 `~/.zshrc`，命令前加：
```bash
export PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH
[ -f ~/.config/adam_tts/secrets.env ] && source ~/.config/adam_tts/secrets.env
```
（adam_tts 自己也会读环境变量，再用 `~/.config/adam_tts/.env` 补充；`--env-file` 可指定。）

## 远程调用（Grok Bot 等不在 Mac mini 上的 agent）
1. 通过面向 Mac mini 的 machine-targeted shell 执行（每条命令需用户批准，**把多步合成一条命令**）。
2. 输入放 `~/adam_tts_jobs/<job>/lines.tsv`（可用 CopyFromBox 传过去），输出到 `~/adam_tts_jobs/<job>/out_<backend>/`。
3. 长任务后台跑并轮询日志（qwen 约 2 倍实时：55 句 ≈ 5 分钟音频 ≈ 13 分钟）：
```bash
export PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH; cd ~/adam_tts_jobs/<job>
nohup adam_tts synth lines.tsv -o out_qwen --backend qwen --voice Serena --style "…" --json > synth.json 2> synth.log &
tail -n 20 synth.log        # 每 2–4 分钟看一次；结束后读 synth.json，再跑 status
```
4. 用 CopyToBox 把 `full.wav`、`full.srt`（需要时 `lines/*.wav`、`manifest.json`）拷回自己的 box。

## 常见问题
- **ffmpeg**：不需要（音频用 soundfile 读写，能解 mp3）；doctor 里 ffmpeg 只是警告。
- **密钥**：任何命令都不会打印密钥值，`doctor` 只显示来源（env / .env / missing）。不要把密钥写进命令、日志或 TSV。
- **qwen 和对齐器**：在 `~/qwen3-tts` 下各自独立的 venv（`.venv`、`aligner/.venv`，transformers 版本冲突无法合并）里以子进程运行，模型每条命令加载一次；路径用 `ADAM_TTS_QWEN_HOME` 改。`doctor --deep --require-aligner` 检查全部。
- **切点不准/weak_cuts**：先听该行；可 `--scene X --force` 重新合成一次，或在台本里给句末加标点。
- **第一次跑 qwen 慢**：模型加载约 10–20 s，之后每个场景约 1–2 倍音频时长。
