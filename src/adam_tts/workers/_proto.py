# Shared by worker scripts (copied inline semantics; imported via sys.path tweak in each worker).
import json
import os
import sys
import traceback


def serve(handlers: dict) -> None:
    proto = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)  # model libraries print to stdout; keep the protocol channel clean
    sys.stdout = sys.stderr

    def send(obj):
        proto.write(json.dumps(obj, ensure_ascii=False) + "\n")
        proto.flush()

    send({"ok": True, "op": "hello", "python": sys.version.split()[0]})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "quit":
                send({"ok": True})
                break
            if op not in handlers:
                send({"ok": False, "type": "UnknownOp", "error": f"unknown op {op}"})
                continue
            send({"ok": True, **(handlers[op](req) or {})})
        except Exception as e:  # noqa: BLE001
            send({"ok": False, "type": type(e).__name__, "error": f"{type(e).__name__}: {e}",
                  "trace": traceback.format_exc()[-3000:]})
