"""lines.tsv: columns id, scene, text (UTF-8, tab-separated, optional header row)."""

from __future__ import annotations

import csv
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from adam_tts.errors import UsageError


@dataclass(frozen=True)
class Line:
    id: str
    scene: str
    text: str


def read_lines(path: Path) -> list[Line]:
    path = Path(path).expanduser()
    if not path.is_file():
        raise UsageError(f"找不到 TSV: {path}")
    rows: list[Line] = []
    seen: dict[str, int] = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for n, r in enumerate(csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE), 1):
            if not r or not any(c.strip() for c in r) or r[0].lstrip().startswith("#"):
                continue
            if n == 1 and [c.strip().lower() for c in r[:3]] == ["id", "scene", "text"]:
                continue
            if len(r) < 3:
                raise UsageError(f"{path}:{n}: 需要 3 列 id<TAB>scene<TAB>text，实际 {len(r)} 列")
            lid, scene, text = r[0].strip(), r[1].strip(), "\t".join(r[2:]).strip()
            if not lid or not scene or not text:
                raise UsageError(f"{path}:{n}: id / scene / text 不能为空")
            if lid in seen:
                raise UsageError(f"{path}:{n}: id 重复: {lid}（首次出现在第 {seen[lid]} 行）")
            seen[lid] = n
            rows.append(Line(lid, scene, text))
    if not rows:
        raise UsageError(f"{path}: 没有任何行")
    # scenes must be contiguous
    order, last = [], None
    for ln in rows:
        if ln.scene != last:
            if ln.scene in order:
                raise UsageError(f"{path}: 场景 {ln.scene} 的行不连续（行 {ln.id}）")
            order.append(ln.scene)
            last = ln.scene
    return rows


def group_scenes(lines: list[Line]) -> "OrderedDict[str, list[Line]]":
    out: OrderedDict[str, list[Line]] = OrderedDict()
    for ln in lines:
        out.setdefault(ln.scene, []).append(ln)
    return out
