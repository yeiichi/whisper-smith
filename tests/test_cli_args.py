from pathlib import Path

import pytest

from whisper_smith.cli import SUPPORTED_TIMESTAMP_FORMATS, infer_output_format, main


def test_cli_help_does_not_call_api(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    assert "usage:" in captured.out.lower()


def test_cli_rejects_missing_audio_file(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["does-not-exist.mp3"])

    assert exc_info.value.code == 2

    captured = capsys.readouterr()
    assert "Audio file not found" in captured.err


def test_infer_output_format_explicit_format_wins() -> None:
    resolved = infer_output_format(Path("sample.txt"), "json")

    assert resolved == "json"


def test_infer_output_format_txt_suffix() -> None:
    resolved = infer_output_format(Path("sample.txt"), None)

    assert resolved == "txt"


def test_infer_output_format_json_suffix() -> None:
    resolved = infer_output_format(Path("sample.json"), None)

    assert resolved == "json"


def test_infer_output_format_srt_suffix() -> None:
    resolved = infer_output_format(Path("sample.srt"), None)

    assert resolved == "srt"


def test_infer_output_format_vtt_suffix() -> None:
    resolved = infer_output_format(Path("sample.vtt"), None)

    assert resolved == "vtt"


def test_infer_output_format_missing_output_defaults_to_txt() -> None:
    resolved = infer_output_format(None, None)

    assert resolved == "txt"


def test_infer_output_format_unsupported_suffix_defaults_to_txt() -> None:
    resolved = infer_output_format(Path("sample.unsupported"), None)

    assert resolved == "txt"


def test_infer_output_format_is_case_insensitive() -> None:
    resolved = infer_output_format(Path("sample.JSON"), None)

    assert resolved == "json"


def test_supported_timestamp_formats() -> None:
    assert SUPPORTED_TIMESTAMP_FORMATS == ("seconds", "hms")
