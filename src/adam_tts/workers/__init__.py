"""Out-of-process workers for heavy local models.

Worker scripts are standalone (stdlib + their model package only) so they can run under ANY python
interpreter, e.g. the existing ~/qwen3-tts/.venv (qwen-tts) and ~/qwen3-tts/aligner/.venv (qwen-asr),
whose transformers pins differ. Protocol: one JSON object per line on stdin -> one JSON per line on
the worker's original stdout (the worker redirects its own stdout prints to stderr).
"""

from pathlib import Path

WORKER_DIR = Path(__file__).resolve().parent
