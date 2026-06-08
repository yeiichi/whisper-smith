import json
from dataclasses import asdict

from whisper_smith.models import TranscriptResult, TranscriptSegment

SUPPORTED_FORMATS = ("txt", "md", "json", "srt", "vtt")
SUPPORTED_TIMESTAMP_FORMATS = ("seconds", "hms")


def export_txt(transcript: TranscriptResult) -> str:
    lines = [_format_speaker_line(segment) for segment in transcript.segments]
    return "\n".join(lines) + "\n"


def export_json(transcript: TranscriptResult, timestamp_format: str = "seconds") -> str:
    normalized_timestamp_format = timestamp_format.lower()
    if normalized_timestamp_format not in SUPPORTED_TIMESTAMP_FORMATS:
        supported = ", ".join(SUPPORTED_TIMESTAMP_FORMATS)
        raise ValueError(
            f"Unsupported timestamp format: {timestamp_format!r}. "
            f"Supported formats: {supported}"
        )

    payload = asdict(transcript)
    if normalized_timestamp_format == "hms":
        for segment in payload.get("segments", []):
            segment["start"] = _format_hms_time(float(segment["start"]))
            segment["end"] = _format_hms_time(float(segment["end"]))

    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    ) + "\n"


def export_srt(transcript: TranscriptResult) -> str:
    blocks: list[str] = []

    for index, segment in enumerate(transcript.segments, start=1):
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{_format_srt_time(segment.start)} --> {_format_srt_time(segment.end)}",
                    _format_speaker_line(segment),
                ]
            )
        )

    return "\n\n".join(blocks) + "\n"


def export_vtt(transcript: TranscriptResult) -> str:
    blocks = ["WEBVTT"]

    for segment in transcript.segments:
        blocks.append(
            "\n".join(
                [
                    f"{_format_vtt_time(segment.start)} --> {_format_vtt_time(segment.end)}",
                    _format_speaker_line(segment),
                ]
            )
        )

    return "\n\n".join(blocks) + "\n"


def export_md(transcript: TranscriptResult) -> str:
    lines = ["# Transcript", ""]

    for segment in transcript.segments:
        speaker = segment.speaker or "UNKNOWN"
        lines.extend(
            [
                f"**{speaker}**",
                "",
                segment.text,
                "",
            ]
        )

    if lines[-1] == "":
        lines.pop()

    return "\n".join(lines) + "\n"


def export_transcript(
    transcript: TranscriptResult,
    output_format: str,
    timestamp_format: str = "seconds",
) -> str:
    normalized_format = output_format.lower().lstrip(".")

    if normalized_format == "txt":
        return export_txt(transcript)

    if normalized_format == "md":
        return export_md(transcript)

    if normalized_format == "json":
        return export_json(transcript, timestamp_format=timestamp_format)

    if normalized_format == "srt":
        return export_srt(transcript)

    if normalized_format == "vtt":
        return export_vtt(transcript)

    supported = ", ".join(SUPPORTED_FORMATS)
    raise ValueError(
        f"Unsupported output format: {output_format!r}. "
        f"Supported formats: {supported}"
    )


def _format_speaker_line(segment: TranscriptSegment) -> str:
    if segment.speaker:
        return f"{segment.speaker}: {segment.text}"

    return segment.text


def _format_srt_time(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    whole_seconds = int(seconds)
    milliseconds = int(round((seconds - whole_seconds) * 1000))

    return f"{int(hours):02}:{int(minutes):02}:{whole_seconds:02},{milliseconds:03}"


def _format_vtt_time(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    whole_seconds = int(seconds)
    milliseconds = int(round((seconds - whole_seconds) * 1000))

    return f"{int(hours):02}:{int(minutes):02}:{whole_seconds:02}.{milliseconds:03}"


def _format_hms_time(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours:02}:{minutes:02}:{secs:02}"
