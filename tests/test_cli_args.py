from pathlib import Path

import pytest

from whisper_smith.models import (
    DiarizationResult,
    DiarizationSegment,
    TranscriptResult,
    TranscriptSegment,
)
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


def test_cli_diarize_defaults_to_json_stdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio_file = tmp_path / "sample.mp3"
    audio_file.write_bytes(b"audio")

    monkeypatch.setattr(
        "whisper_smith.cli.diarize_audio",
        lambda *_args, **_kwargs: DiarizationResult(
            segments=[
                DiarizationSegment(start=0.0, end=1.25, speaker="SPEAKER_00"),
            ]
        ),
    )
    monkeypatch.setattr(
        "whisper_smith.cli.transcribe_audio",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("Transcription should not run for --diarize")
        ),
    )

    main([str(audio_file), "--diarize"])

    captured = capsys.readouterr()
    assert '"speaker": "SPEAKER_00"' in captured.out
    assert '"end": 1.25' in captured.out


def test_cli_diarize_rejects_non_json_format(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio_file = tmp_path / "sample.mp3"
    audio_file.write_bytes(b"audio")

    with pytest.raises(SystemExit) as exc_info:
        main([str(audio_file), "--diarize", "--format", "txt"])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "Diarization output currently supports only JSON" in captured.err


def test_cli_diarize_rejects_conflicting_speaker_options(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio_file = tmp_path / "sample.mp3"
    audio_file.write_bytes(b"audio")

    with pytest.raises(SystemExit) as exc_info:
        main([str(audio_file), "--diarize", "--num-speakers", "2", "--min-speakers", "1"])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "--num-speakers cannot be combined" in captured.err


def test_cli_diarization_options_require_diarize_or_align(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio_file = tmp_path / "sample.mp3"
    audio_file.write_bytes(b"audio")

    with pytest.raises(SystemExit) as exc_info:
        main([str(audio_file), "--num-speakers", "2"])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "Diarization options require --diarize or --align" in captured.err

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                str(audio_file),
                "--diarization-model",
                "pyannote/speaker-diarization-community-1",
            ]
        )

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "Diarization options require --diarize or --align" in captured.err


def test_cli_diarize_passes_speaker_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_file = tmp_path / "sample.mp3"
    audio_file.write_bytes(b"audio")
    calls = []

    def fake_diarize(*args, **kwargs):
        calls.append((args, kwargs))
        return DiarizationResult(segments=[])

    monkeypatch.setattr("whisper_smith.cli.diarize_audio", fake_diarize)

    main(
        [
            str(audio_file),
            "--diarize",
            "--format",
            "json",
            "--diarization-model",
            "pyannote/speaker-diarization-community-1",
            "--min-speakers",
            "1",
            "--max-speakers",
            "3",
        ]
    )

    assert calls == [
        (
            (audio_file,),
            {
                "model": "pyannote/speaker-diarization-community-1",
                "num_speakers": None,
                "min_speakers": 1,
                "max_speakers": 3,
            },
        )
    ]


def test_cli_align_rejects_non_json_format(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio_file = tmp_path / "sample.mp3"
    audio_file.write_bytes(b"audio")

    with pytest.raises(SystemExit) as exc_info:
        main([str(audio_file), "--align", "--format", "txt"])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "Aligned transcript output currently supports only JSON" in captured.err


def test_cli_align_passes_speaker_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_file = tmp_path / "sample.mp3"
    audio_file.write_bytes(b"audio")
    calls = []

    monkeypatch.setattr(
        "whisper_smith.cli.transcribe_audio",
        lambda _path: TranscriptResult(segments=[]),
    )

    def fake_diarize(*args, **kwargs):
        calls.append((args, kwargs))
        return DiarizationResult(segments=[])

    monkeypatch.setattr("whisper_smith.cli.diarize_audio", fake_diarize)

    main(
        [
            str(audio_file),
            "--align",
            "--format",
            "json",
            "--diarization-model",
            "pyannote/speaker-diarization-community-1",
            "--min-speakers",
            "1",
            "--max-speakers",
            "3",
        ]
    )

    assert calls == [
        (
            (audio_file,),
            {
                "model": "pyannote/speaker-diarization-community-1",
                "num_speakers": None,
                "min_speakers": 1,
                "max_speakers": 3,
            },
        )
    ]


def test_cli_align_rejects_diarize_mode_combination(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio_file = tmp_path / "sample.mp3"
    audio_file.write_bytes(b"audio")

    with pytest.raises(SystemExit) as exc_info:
        main([str(audio_file), "--align", "--diarize"])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "--diarize cannot be combined with --align" in captured.err


def test_cli_artifacts_dir_requires_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio_file = tmp_path / "sample.mp3"
    audio_file.write_bytes(b"audio")

    with pytest.raises(SystemExit) as exc_info:
        main([str(audio_file), "--align", "--artifacts-dir", str(tmp_path)])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "--artifacts-dir requires --output" in captured.err


def test_cli_align_stdout_writes_aligned_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio_file = tmp_path / "sample.mp3"
    audio_file.write_bytes(b"audio")

    monkeypatch.setattr(
        "whisper_smith.cli.transcribe_audio",
        lambda _path: TranscriptResult(
            segments=[
                TranscriptSegment(start=0.0, end=1.0, text="Hello."),
            ]
        ),
    )
    monkeypatch.setattr(
        "whisper_smith.cli.diarize_audio",
        lambda *_args, **_kwargs: DiarizationResult(
            segments=[
                DiarizationSegment(start=0.0, end=1.0, speaker="SPEAKER_00"),
            ]
        ),
    )

    main([str(audio_file), "--align"])

    captured = capsys.readouterr()
    assert '"speaker": "SPEAKER_00"' in captured.out
    assert '"text": "Hello."' in captured.out
