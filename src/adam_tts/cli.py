"""adam_tts command line. argparse: zero deps, full control over exit codes and --json errors."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from adam_tts import __version__, output, service
from adam_tts.backends import registry
from adam_tts.errors import EXIT_INTERNAL, AdamTTSError, UsageError


class _Parser(argparse.ArgumentParser):
    def error(self, message):  # route usage errors through our JSON/exit-code path
        raise UsageError(message, hint=f"运行 {self.prog} --help 查看用法")


def _global_opts(p: argparse.ArgumentParser, suppress: bool) -> None:
    d = {"default": argparse.SUPPRESS} if suppress else {}
    p.add_argument("--json", action="store_true", help="stdout 只输出一个 JSON 对象", **d)
    p.add_argument("--config", metavar="PATH", help="config.toml 路径", **({} if not suppress else d))
    p.add_argument("--env-file", metavar="PATH", help=".env 路径（补充环境变量，不覆盖）", **d)
    p.add_argument("-v", "--verbose", action="store_true", help="在 stderr 输出更多日志", **d)


def _synth_opts(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("合成参数（命令行 > 预设 > 后端默认）")
    g.add_argument("--preset", help="预设名（默认 draft）")
    g.add_argument("--backend", choices=registry.names(), help="后端（只给它时不套用预设）")
    g.add_argument("--voice", help="后端原生音色名")
    g.add_argument("--style", help="风格指令（'' 清空预设值）")
    g.add_argument("--prefix", help="行首内联标签，如 '(平静 干练)'（'' 清空）")
    g.add_argument("--speed", type=float, help="语速倍率 0.5–2.0（仅 edge/doubao）")
    g.add_argument("--pitch", type=int, help="音调偏移 Hz（仅 edge）")
    g.add_argument("--volume", type=float, help="音量倍率 0.5–2.0（仅 edge/doubao）")
    g.add_argument("--lang", help="语言（仅 qwen）")


def build_parser() -> argparse.ArgumentParser:
    p = _Parser(prog="adam_tts", description="TTS 工作台（给 agent 用的 CLI）")
    p.add_argument("--version", action="store_true", help="打印版本")
    _global_opts(p, suppress=False)
    sub = p.add_subparsers(dest="command", parser_class=_Parser)

    s = sub.add_parser("say", help="单段文本快速合成到 wav")
    _global_opts(s, suppress=True)
    s.add_argument("text", nargs="?", help="要念的文本")
    s.add_argument("--text-file", help="从文件读文本（UTF-8）")
    s.add_argument("-o", "--out", required=True, help="输出 .wav 路径")
    _synth_opts(s)

    v = sub.add_parser("voice", help="音色相关")
    _global_opts(v, suppress=True)
    vsub = v.add_subparsers(dest="voice_command", parser_class=_Parser)
    vl = vsub.add_parser("list", help="列出后端原生音色")
    _global_opts(vl, suppress=True)
    vl.add_argument("--backend", required=True, choices=registry.names())
    vl.add_argument("--lang", help="按语言前缀过滤，如 zh / zh-CN")

    d = sub.add_parser("doctor", help="检查密钥（不显示值）、后端、ffmpeg")
    _global_opts(d, suppress=True)
    d.add_argument("--backend", choices=registry.names(), help="只检查该后端（作为必需项）")
    d.add_argument("--deep", action="store_true", help="真合成一句短文本（会产生费用）")
    return p


def _human_say(r: dict) -> str:
    return f"{r['out']}  {r['duration_s']:.2f}s  backend={r['backend']} voice={r['voice']} preset={r['preset']}"


def _human_voices(r: dict) -> str:
    lines = [f"{r['backend']}: {r['count']} voices"]
    for v in r["voices"]:
        lines.append("  " + "  ".join(x for x in (v["name"], v.get("lang") or "", v.get("gender") or "",
                                                   v.get("description") or "") if x))
    return "\n".join(lines)


def _human_doctor(r: dict) -> str:
    mark = {"ok": "✓", "warn": "!", "fail": "✗", "skip": "-"}
    lines = ["secrets: " + ", ".join(f"{k}={v}" for k, v in r["secrets"].items())]
    for c in r["checks"]:
        req = " (required)" if c["required"] else ""
        lines.append(f"{mark.get(c['status'], '?')} {c['name']}: {c['detail']}{req}")
    lines.append("healthy" if r["healthy"] else "NOT healthy")
    return "\n".join(lines)


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    j = args.json
    if args.version:
        output.emit({"version": __version__}, j, f"adam_tts {__version__}")
        return 0
    if args.command is None:
        raise UsageError("缺少命令", hint="可用命令: say, voice list, doctor")
    ctx = service.Context(config_path=args.config, env_file=args.env_file)

    if args.command == "say":
        if bool(args.text) == bool(args.text_file):
            raise UsageError("需要且只能给一个：文本参数 或 --text-file")
        text = args.text if args.text else Path(args.text_file).expanduser().read_text(encoding="utf-8")
        params = {k: getattr(args, k) for k in ("voice", "style", "prefix", "speed", "pitch", "volume", "lang")}
        r = service.say(ctx, text=text, out=Path(args.out), preset=args.preset, backend=args.backend,
                        params=params)
        output.emit(r, j, _human_say(r))
        return 0

    if args.command == "voice":
        if args.voice_command != "list":
            raise UsageError("缺少子命令", hint="adam_tts voice list --backend <name>")
        r = service.voice_list(ctx, backend=args.backend, lang=args.lang)
        output.emit(r, j, _human_voices(r))
        return 0

    if args.command == "doctor":
        r, code = service.doctor(ctx, backend=args.backend, deep=args.deep)
        if j:
            import json as _json
            sys.stdout.write(_json.dumps({"ok": code == 0, **r}, ensure_ascii=False) + "\n")
        else:
            output.emit(r, False, _human_doctor(r))
        return code

    raise UsageError(f"未知命令: {args.command}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    json_mode = "--json" in argv
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        return run(args, parser)
    except AdamTTSError as e:
        output.emit_error(e.to_dict(), json_mode)
        return e.exit_code
    except KeyboardInterrupt:
        return 130
    except Exception as e:  # noqa: BLE001 - last-resort handler; never leak secrets
        from adam_tts.secrets import Secrets

        msg = Secrets().redact(f"{type(e).__name__}: {e}")
        output.emit_error({"code": EXIT_INTERNAL, "type": "InternalError", "message": msg, "hint": None},
                          json_mode)
        return EXIT_INTERNAL


if __name__ == "__main__":
    raise SystemExit(main())
