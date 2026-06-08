from __future__ import annotations

from whisper_smith.models import TranscriptResult


def export_md(result: TranscriptResult) -> str:
    lines: list[str] = ["# Transcript", ""]

    if result.segments:
        for segment in result.segments:
            speaker = segment.speaker or "Speaker"
            lines.append(f"**{speaker}**")
            lines.append("")
            lines.append(segment.text.strip())
            lines.append("")

        return "\n".join(lines).strip() + "\n"

    lines.append(result.text.strip())
    return "\n".join(lines).strip() + "\n"