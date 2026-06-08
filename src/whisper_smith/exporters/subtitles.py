from __future__ import annotations

from whisper_smith.models import TranscriptResult


def _format_srt_timestamp(seconds: float) -> str:
    milliseconds_total = int(round(seconds * 1000))

    hours = milliseconds_total // 3_600_000
    minutes = milliseconds_total % 3_600_000 // 60_000
    secs = milliseconds_total % 60_000 // 1_000
    milliseconds = milliseconds_total % 1_000

    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


def _format_vtt_timestamp(seconds: float) -> str:
    milliseconds_total = int(round(seconds * 1000))

    hours = milliseconds_total // 3_600_000
    minutes = milliseconds_total % 3_600_000 // 60_000
    secs = milliseconds_total % 60_000 // 1_000
    milliseconds = milliseconds_total % 1_000

    return f"{hours:02}:{minutes:02}:{secs:02}.{milliseconds:03}"


def export_srt(result: TranscriptResult) -> str:
    blocks: list[str] = []

    for index, segment in enumerate(result.segments, start=1):
        start = _format_srt_timestamp(segment.start)
        end = _format_srt_timestamp(segment.end)
        speaker = f"{segment.speaker}: " if segment.speaker else ""
        text = f"{speaker}{segment.text.strip()}"

        blocks.append(f"{index}\n{start} --> {end}\n{text}")

    return "\n\n".join(blocks).strip() + "\n"


def export_vtt(result: TranscriptResult) -> str:
    blocks: list[str] = ["WEBVTT"]

    for segment in result.segments:
        start = _format_vtt_timestamp(segment.start)
        end = _format_vtt_timestamp(segment.end)
        speaker = f"{segment.speaker}: " if segment.speaker else ""
        text = f"{speaker}{segment.text.strip()}"

        blocks.append(f"{start} --> {end}\n{text}")

    return "\n\n".join(blocks).strip() + "\n"