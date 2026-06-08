from __future__ import annotations

from whisper_smith.models import TranscriptResult


def export_txt(result: TranscriptResult) -> str:
    if result.segments:
        lines: list[str] = []

        for segment in result.segments:
            speaker = f"{segment.speaker}: " if segment.speaker else ""
            lines.append(f"{speaker}{segment.text.strip()}")

        return "\n".join(lines).strip() + "\n"

    return result.text.strip() + "\n"