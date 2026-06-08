from pathlib import Path

import pytest

from whisper_smith.models import (
    DiarizationResult,
    DiarizationSegment,
    TranscriptResult,
    TranscriptSegment,
)
from whisper_smith.cli import (
    main,
    resolve_alignment_artifact_paths,
    resolve_output_path,
)


def test_resolve_output_path_keeps_explicit_file_path() -> None:
    resolved = resolve_output_path(
        audio_path=Path("data/sample.mp3"),
        output_path=Path("outputs/result.txt"),
        output_format="txt",
    )

    assert resolved == Path("outputs/result.txt")


def test_resolve_output_path_treats_trailing_slash_as_directory() -> None:
    resolved = resolve_output_path(
        audio_path=Path("data/sample.mp3"),
        output_path=Path("outputs"),
        output_format="txt",
        raw_output_arg="outputs/",
    )

    assert resolved == Path("outputs/sample.txt")


def test_resolve_output_path_treats_existing_directory_without_slash_as_directory(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    resolved = resolve_output_path(
        audio_path=Path("data/sample.mp3"),
        output_path=output_dir,
        output_format="txt",
    )

    assert resolved == output_dir / "sample.txt"


def test_resolve_output_path_treats_non_existing_path_without_slash_as_file() -> None:
    resolved = resolve_output_path(
        audio_path=Path("data/sample.mp3"),
        output_path=Path("outputs"),
        output_format="txt",
    )

    assert resolved == Path("outputs")


def test_cli_creates_parent_directory_before_writing_output(
    tmp_path: Path, monkeypatch
) -> None:
    audio_file = tmp_path / "sample.mp3"
    audio_file.write_bytes(b"audio")
    output_file = tmp_path / "outputs" / "result.txt"

    monkeypatch.setattr("whisper_smith.cli.transcribe_audio", lambda _: "TRANSCRIPT")
    monkeypatch.setattr("whisper_smith.cli.export_transcript", lambda *_, **__: "rendered")

    main([str(audio_file), "--output", str(output_file)])

    assert output_file.exists()
    assert output_file.read_text(encoding="utf-8") == "rendered"


def test_cli_refuses_to_overwrite_existing_output_file_without_flag(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    audio_file = tmp_path / "sample.mp3"
    audio_file.write_bytes(b"audio")
    output_file = tmp_path / "result.txt"
    output_file.write_text("existing", encoding="utf-8")

    transcribe_called = False

    def fake_transcribe(_: Path) -> str:
        nonlocal transcribe_called
        transcribe_called = True
        return "TRANSCRIPT"

    monkeypatch.setattr("whisper_smith.cli.transcribe_audio", fake_transcribe)
    monkeypatch.setattr("whisper_smith.cli.export_transcript", lambda *_, **__: "rendered")

    with pytest.raises(SystemExit) as exc_info:
        main([str(audio_file), "--output", str(output_file)])

    assert exc_info.value.code == 2
    assert transcribe_called is False
    assert output_file.read_text(encoding="utf-8") == "existing"

    captured = capsys.readouterr()
    assert "Output file already exists" in captured.err
    assert "--overwrite" in captured.err


def test_cli_overwrites_existing_output_file_with_flag(
    tmp_path: Path, monkeypatch
) -> None:
    audio_file = tmp_path / "sample.mp3"
    audio_file.write_bytes(b"audio")
    output_file = tmp_path / "result.txt"
    output_file.write_text("existing", encoding="utf-8")

    monkeypatch.setattr("whisper_smith.cli.transcribe_audio", lambda _: "TRANSCRIPT")
    monkeypatch.setattr("whisper_smith.cli.export_transcript", lambda *_, **__: "rendered")

    main([str(audio_file), "--output", str(output_file), "--overwrite"])

    assert output_file.read_text(encoding="utf-8") == "rendered"


def test_cli_writes_new_output_file_without_overwrite_flag(
    tmp_path: Path, monkeypatch
) -> None:
    audio_file = tmp_path / "sample.mp3"
    audio_file.write_bytes(b"audio")
    output_file = tmp_path / "result.txt"

    monkeypatch.setattr("whisper_smith.cli.transcribe_audio", lambda _: "TRANSCRIPT")
    monkeypatch.setattr("whisper_smith.cli.export_transcript", lambda *_, **__: "rendered")

    main([str(audio_file), "--output", str(output_file)])

    assert output_file.read_text(encoding="utf-8") == "rendered"


def test_cli_refuses_to_overwrite_resolved_output_file_in_directory_without_flag(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    audio_file = tmp_path / "sample.mp3"
    audio_file.write_bytes(b"audio")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    resolved_output = output_dir / "sample.txt"
    resolved_output.write_text("existing", encoding="utf-8")

    transcribe_called = False

    def fake_transcribe(_: Path) -> str:
        nonlocal transcribe_called
        transcribe_called = True
        return "TRANSCRIPT"

    monkeypatch.setattr("whisper_smith.cli.transcribe_audio", fake_transcribe)
    monkeypatch.setattr("whisper_smith.cli.export_transcript", lambda *_, **__: "rendered")

    with pytest.raises(SystemExit) as exc_info:
        main([str(audio_file), "--output", str(output_dir) + "/"])

    assert exc_info.value.code == 2
    assert transcribe_called is False
    assert resolved_output.read_text(encoding="utf-8") == "existing"

    captured = capsys.readouterr()
    assert "Output file already exists" in captured.err
    assert str(resolved_output) in captured.err


def test_cli_diarize_writes_json_output_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_file = tmp_path / "sample.mp3"
    audio_file.write_bytes(b"audio")
    output_file = tmp_path / "diarization.json"

    monkeypatch.setattr(
        "whisper_smith.cli.diarize_audio",
        lambda *_args, **_kwargs: DiarizationResult(
            segments=[
                DiarizationSegment(start=0.0, end=1.0, speaker="SPEAKER_00"),
            ]
        ),
    )

    main([str(audio_file), "--diarize", "--output", str(output_file)])

    output = output_file.read_text(encoding="utf-8")
    assert '"speaker": "SPEAKER_00"' in output
    assert '"segments"' in output


def test_cli_diarize_directory_output_uses_json_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_file = tmp_path / "sample.mp3"
    audio_file.write_bytes(b"audio")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    monkeypatch.setattr(
        "whisper_smith.cli.diarize_audio",
        lambda *_args, **_kwargs: DiarizationResult(segments=[]),
    )

    main([str(audio_file), "--diarize", "--output", str(output_dir) + "/"])

    assert (output_dir / "sample.json").exists()


def test_resolve_alignment_artifact_paths_defaults_to_aligned_output_directory() -> None:
    transcript_output, diarization_output = resolve_alignment_artifact_paths(
        audio_path=Path("data/sample.mp3"),
        aligned_output_path=Path("outputs/sample.aligned.json"),
        artifacts_dir=None,
    )

    assert transcript_output == Path("outputs/sample.transcript.json")
    assert diarization_output == Path("outputs/sample.diarization.json")


def test_resolve_alignment_artifact_paths_uses_custom_artifacts_directory() -> None:
    transcript_output, diarization_output = resolve_alignment_artifact_paths(
        audio_path=Path("data/sample.mp3"),
        aligned_output_path=Path("outputs/sample.aligned.json"),
        artifacts_dir=Path("artifacts"),
    )

    assert transcript_output == Path("artifacts/sample.transcript.json")
    assert diarization_output == Path("artifacts/sample.diarization.json")


def test_cli_align_writes_aligned_json_and_intermediate_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_file = tmp_path / "sample.mp3"
    audio_file.write_bytes(b"audio")
    output_file = tmp_path / "outputs" / "sample.aligned.json"

    monkeypatch.setattr(
        "whisper_smith.cli.transcribe_audio",
        lambda _path: TranscriptResult(
            segments=[
                TranscriptSegment(start=0.0, end=1.0, text="Hello."),
            ],
        ),
    )
    monkeypatch.setattr(
        "whisper_smith.cli.diarize_audio",
        lambda *_args, **_kwargs: DiarizationResult(
            segments=[
                DiarizationSegment(start=0.0, end=1.0, speaker="SPEAKER_00"),
            ],
        ),
    )

    main([str(audio_file), "--align", "--output", str(output_file)])

    assert '"speaker": "SPEAKER_00"' in output_file.read_text(encoding="utf-8")
    transcript_output = tmp_path / "outputs" / "sample.transcript.json"
    diarization_output = tmp_path / "outputs" / "sample.diarization.json"
    assert '"speaker": null' in transcript_output.read_text(encoding="utf-8")
    assert '"speaker": "SPEAKER_00"' in diarization_output.read_text(encoding="utf-8")


def test_cli_align_directory_output_uses_aligned_json_suffix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_file = tmp_path / "sample.mp3"
    audio_file.write_bytes(b"audio")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    monkeypatch.setattr(
        "whisper_smith.cli.transcribe_audio",
        lambda _path: TranscriptResult(segments=[]),
    )
    monkeypatch.setattr(
        "whisper_smith.cli.diarize_audio",
        lambda *_args, **_kwargs: DiarizationResult(segments=[]),
    )

    main([str(audio_file), "--align", "--output", str(output_dir) + "/"])

    assert (output_dir / "sample.aligned.json").exists()
    assert (output_dir / "sample.transcript.json").exists()
    assert (output_dir / "sample.diarization.json").exists()


def test_cli_align_writes_artifacts_to_custom_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_file = tmp_path / "sample.mp3"
    audio_file.write_bytes(b"audio")
    output_file = tmp_path / "outputs" / "sample.aligned.json"
    artifacts_dir = tmp_path / "artifacts"

    monkeypatch.setattr(
        "whisper_smith.cli.transcribe_audio",
        lambda _path: TranscriptResult(segments=[]),
    )
    monkeypatch.setattr(
        "whisper_smith.cli.diarize_audio",
        lambda *_args, **_kwargs: DiarizationResult(segments=[]),
    )

    main(
        [
            str(audio_file),
            "--align",
            "--output",
            str(output_file),
            "--artifacts-dir",
            str(artifacts_dir),
        ]
    )

    assert output_file.exists()
    assert (artifacts_dir / "sample.transcript.json").exists()
    assert (artifacts_dir / "sample.diarization.json").exists()
    assert not (output_file.parent / "sample.transcript.json").exists()


def test_cli_align_no_artifacts_writes_only_aligned_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_file = tmp_path / "sample.mp3"
    audio_file.write_bytes(b"audio")
    output_file = tmp_path / "outputs" / "sample.aligned.json"

    monkeypatch.setattr(
        "whisper_smith.cli.transcribe_audio",
        lambda _path: TranscriptResult(segments=[]),
    )
    monkeypatch.setattr(
        "whisper_smith.cli.diarize_audio",
        lambda *_args, **_kwargs: DiarizationResult(segments=[]),
    )

    main([str(audio_file), "--align", "--output", str(output_file), "--no-artifacts"])

    assert output_file.exists()
    assert not (output_file.parent / "sample.transcript.json").exists()
    assert not (output_file.parent / "sample.diarization.json").exists()


def test_cli_align_refuses_to_overwrite_existing_artifact_without_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    audio_file = tmp_path / "sample.mp3"
    audio_file.write_bytes(b"audio")
    output_file = tmp_path / "outputs" / "sample.aligned.json"
    artifact_file = tmp_path / "outputs" / "sample.transcript.json"
    artifact_file.parent.mkdir()
    artifact_file.write_text("existing", encoding="utf-8")
    transcribe_called = False

    def fake_transcribe(_: Path) -> TranscriptResult:
        nonlocal transcribe_called
        transcribe_called = True
        return TranscriptResult(segments=[])

    monkeypatch.setattr("whisper_smith.cli.transcribe_audio", fake_transcribe)

    with pytest.raises(SystemExit) as exc_info:
        main([str(audio_file), "--align", "--output", str(output_file)])

    assert exc_info.value.code == 2
    assert transcribe_called is False
    captured = capsys.readouterr()
    assert "Output file already exists" in captured.err
    assert str(artifact_file) in captured.err
