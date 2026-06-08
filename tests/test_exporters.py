from whisper_smith.exporters import (
    export_json,
    export_md,
    export_srt,
    export_txt,
    export_vtt,
)
from whisper_smith.models import TranscriptResult, TranscriptSegment


def make_result() -> TranscriptResult:
    return TranscriptResult(
        text="Hello world. This is a test.",
        segments=[
            TranscriptSegment(
                start=0.0,
                end=1.5,
                text="Hello world.",
                speaker="SPEAKER_00",
            ),
            TranscriptSegment(
                start=1.5,
                end=3.0,
                text="This is a test.",
                speaker="SPEAKER_01",
            ),
        ],
    )


def test_export_txt() -> None:
    result = make_result()

    assert export_txt(result) == (
        "SPEAKER_00: Hello world.\n"
        "SPEAKER_01: This is a test.\n"
    )


def test_export_json() -> None:
    result = make_result()

    output = export_json(result)

    assert '"text": "Hello world. This is a test."' in output
    assert '"speaker": "SPEAKER_00"' in output
    assert '"start": 0.0' in output


def test_export_json_hms_timestamps() -> None:
    result = make_result()

    output = export_json(result, timestamp_format="hms")

    assert '"start": "00:00:00"' in output
    assert '"end": "00:00:02"' in output


def test_export_srt() -> None:
    result = make_result()

    assert export_srt(result) == (
        "1\n"
        "00:00:00,000 --> 00:00:01,500\n"
        "SPEAKER_00: Hello world.\n"
        "\n"
        "2\n"
        "00:00:01,500 --> 00:00:03,000\n"
        "SPEAKER_01: This is a test.\n"
    )


def test_export_vtt() -> None:
    result = make_result()

    assert export_vtt(result) == (
        "WEBVTT\n"
        "\n"
        "00:00:00.000 --> 00:00:01.500\n"
        "SPEAKER_00: Hello world.\n"
        "\n"
        "00:00:01.500 --> 00:00:03.000\n"
        "SPEAKER_01: This is a test.\n"
    )


def test_export_md() -> None:
    result = make_result()

    assert export_md(result) == (
        "# Transcript\n"
        "\n"
        "**SPEAKER_00**\n"
        "\n"
        "Hello world.\n"
        "\n"
        "**SPEAKER_01**\n"
        "\n"
        "This is a test.\n"
    )
