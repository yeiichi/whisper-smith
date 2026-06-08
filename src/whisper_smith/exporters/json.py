from __future__ import annotations

import json
from dataclasses import asdict

from whisper_smith.models import TranscriptResult


def export_json(result: TranscriptResult) -> str:
    return json.dumps(
        asdict(result),
        ensure_ascii=False,
        indent=2,
    ) + "\n"