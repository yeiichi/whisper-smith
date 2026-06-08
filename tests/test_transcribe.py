from types import SimpleNamespace

import httpx
import pytest

from whisper_smith.models import TranscriptResult
from whisper_smith.transcribe import (
    from_openai_response,
    transcribe_audio,
)
from openai import BadRequestError


def test_from_openai_response_returns_transcript() -> None:
    response = SimpleNamespace(text="Hello world.")

    transcript = from_openai_response(response)

    assert isinstance(transcript, TranscriptResult)
    assert transcript.text == "Hello world."
    assert len(transcript.segments) == 1
    assert transcript.segments[0].start == 0.0
    assert transcript.segments[0].end == 0.0
    assert transcript.segments[0].speaker is None


def test_from_openai_response_uses_timestamped_segments_and_speaker() -> None:
    response = SimpleNamespace(
        text="Hello\nWorld",
        segments=[
            SimpleNamespace(start=0.1, end=1.2, text="Hello", speaker="SPEAKER_00"),
            SimpleNamespace(start=1.3, end=2.4, text="World", speaker="SPEAKER_01"),
        ],
    )

    transcript = from_openai_response(response)

    assert len(transcript.segments) == 2
    assert transcript.segments[0].start == 0.1
    assert transcript.segments[0].end == 1.2
    assert transcript.segments[0].speaker == "SPEAKER_00"
    assert transcript.segments[1].speaker == "SPEAKER_01"
    assert transcript.text == "Hello\nWorld"


def test_transcribe_audio_raises_for_missing_file(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called() -> None:
        raise AssertionError("OpenAI client should not be created for missing files")

    monkeypatch.setattr("whisper_smith.transcribe.OpenAI", fail_if_called)

    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        transcribe_audio("does-not-exist.mp3")


def test_transcribe_audio_uses_chunking_for_large_files(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "large.m4a"
    audio_path.write_bytes(b"dummy audio")

    monkeypatch.setattr("whisper_smith.transcribe.MAX_SINGLE_UPLOAD_BYTES", 1)
    monkeypatch.setattr(
        "whisper_smith.transcribe._transcribe_in_chunks",
        lambda path, openai_client, model: TranscriptResult(
            segments=[],
            text="chunked transcript",
        ),
    )
    monkeypatch.setattr(
        "whisper_smith.transcribe._transcribe_single_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Single-file path should not run for large files")
        ),
    )
    monkeypatch.setattr("whisper_smith.transcribe.OpenAI", lambda: object())

    result = transcribe_audio(audio_path)
    assert result.text == "chunked transcript"


def test_transcribe_audio_uses_timestamp_model_by_default(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "small.m4a"
    audio_path.write_bytes(b"dummy audio")
    calls = []

    def fake_single_file(path, openai_client, model):
        calls.append((path, openai_client, model))
        return TranscriptResult(
            segments=[
                SimpleNamespace(start=0.5, end=1.5, text="timestamped", speaker=None)
            ],
            text="timestamped",
        )

    monkeypatch.setattr("whisper_smith.transcribe._transcribe_single_file", fake_single_file)
    monkeypatch.setattr("whisper_smith.transcribe.OpenAI", lambda: object())

    result = transcribe_audio(audio_path)

    assert result.text == "timestamped"
    assert calls[0][2] == "whisper-1"


def test_transcribe_audio_can_opt_out_of_timestamp_model(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "small.m4a"
    audio_path.write_bytes(b"dummy audio")
    calls = []

    def fake_single_file(path, openai_client, model):
        calls.append((path, openai_client, model))
        return TranscriptResult(
            segments=[SimpleNamespace(start=0.0, end=0.0, text="plain", speaker=None)],
            text="plain",
        )

    monkeypatch.setattr("whisper_smith.transcribe._transcribe_single_file", fake_single_file)
    monkeypatch.setattr("whisper_smith.transcribe.OpenAI", lambda: object())

    result = transcribe_audio(audio_path, require_timestamps=False)

    assert result.text == "plain"
    assert calls[0][2] == "gpt-4o-transcribe"


def test_transcribe_audio_falls_back_to_chunking_on_bad_request(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "small.m4a"
    audio_path.write_bytes(b"dummy audio")

    request = httpx.Request("POST", "https://api.openai.com/v1/audio/transcriptions")
    response = httpx.Response(400, request=request)
    bad_request = BadRequestError(
        "invalid file",
        response=response,
        body={
            "error": {
                "message": "Audio file might be corrupted or unsupported",
                "type": "invalid_request_error",
                "param": "file",
                "code": "invalid_value",
            }
        },
    )

    monkeypatch.setattr(
        "whisper_smith.transcribe._transcribe_single_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(bad_request),
    )
    monkeypatch.setattr(
        "whisper_smith.transcribe._transcribe_in_chunks",
        lambda path, openai_client, model: TranscriptResult(
            segments=[],
            text="fallback chunk transcript",
        ),
    )
    monkeypatch.setattr("whisper_smith.transcribe.OpenAI", lambda: object())

    result = transcribe_audio(audio_path)
    assert result.text == "fallback chunk transcript"


def test_transcribe_single_file_falls_back_when_verbose_json_unsupported(
    tmp_path,
) -> None:
    audio_path = tmp_path / "small.m4a"
    audio_path.write_bytes(b"dummy audio")

    request = httpx.Request("POST", "https://api.openai.com/v1/audio/transcriptions")
    response_400 = httpx.Response(400, request=request)
    unsupported_verbose = BadRequestError(
        "unsupported response format",
        response=response_400,
        body={
            "error": {
                "message": "response_format 'verbose_json' is not compatible with this model",
                "type": "invalid_request_error",
                "param": "response_format",
                "code": "unsupported_value",
            }
        },
    )

    class FakeTranscriptions:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise unsupported_verbose
            return SimpleNamespace(text="plain json transcript")

    fake_client = SimpleNamespace(
        audio=SimpleNamespace(transcriptions=FakeTranscriptions())
    )

    from whisper_smith.transcribe import _transcribe_single_file

    result = _transcribe_single_file(audio_path, fake_client, "test-model")
    assert result.text == "plain json transcript"
    assert len(result.segments) == 1


def test_is_response_format_unsupported_matches_exception_text_when_body_missing() -> None:
    request = httpx.Request("POST", "https://api.openai.com/v1/audio/transcriptions")
    response_400 = httpx.Response(400, request=request)
    error = BadRequestError(
        "response_format 'verbose_json' is not compatible with model",
        response=response_400,
        body=None,
    )

    from whisper_smith.transcribe import _is_response_format_unsupported

    assert _is_response_format_unsupported(error) is True


def test_resolve_ffmpeg_executable_prefers_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("whisper_smith.transcribe.shutil.which", lambda _: "/usr/bin/ffmpeg")
    from whisper_smith.transcribe import _resolve_ffmpeg_executable

    assert _resolve_ffmpeg_executable() == "/usr/bin/ffmpeg"


def test_transcribe_in_chunks_offsets_segment_times(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "large.m4a"
    audio_path.write_bytes(b"dummy audio")

    chunk_a = tmp_path / "chunk_00000.m4a"
    chunk_b = tmp_path / "chunk_00001.m4a"
    chunk_a.write_bytes(b"a")
    chunk_b.write_bytes(b"b")

    monkeypatch.setattr(
        "whisper_smith.transcribe._split_audio_into_chunks",
        lambda *_: [chunk_a, chunk_b],
    )

    def fake_single_file(path, *_args, **_kwargs):
        if path == chunk_a:
            return TranscriptResult(
                segments=[SimpleNamespace(start=1.0, end=2.0, text="first", speaker=None)],
                text="first",
            )
        return TranscriptResult(
            segments=[SimpleNamespace(start=2.5, end=3.5, text="second", speaker="S1")],
            text="second",
        )

    monkeypatch.setattr("whisper_smith.transcribe._transcribe_single_file", fake_single_file)

    from whisper_smith.transcribe import _transcribe_in_chunks

    result = _transcribe_in_chunks(audio_path, openai_client=object(), model="test-model")

    assert len(result.segments) == 2
    assert result.segments[0].start == 1.0
    assert result.segments[0].end == 2.0
    assert result.segments[1].start == 62.5
    assert result.segments[1].end == 63.5
    assert result.segments[1].speaker == "S1"
    assert result.text == "first\nsecond"


def test_transcribe_in_chunks_sets_end_from_chunk_duration_when_zero_length_segment(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "large.m4a"
    audio_path.write_bytes(b"dummy audio")

    chunk_a = tmp_path / "chunk_00000.m4a"
    chunk_b = tmp_path / "chunk_00001.m4a"
    chunk_a.write_bytes(b"a")
    chunk_b.write_bytes(b"b")

    monkeypatch.setattr(
        "whisper_smith.transcribe._split_audio_into_chunks",
        lambda *_: [chunk_a, chunk_b],
    )
    monkeypatch.setattr(
        "whisper_smith.transcribe._get_audio_duration_seconds",
        lambda chunk: 59.4 if chunk == chunk_a else 34.2,
    )
    monkeypatch.setattr(
        "whisper_smith.transcribe._transcribe_single_file",
        lambda *args, **kwargs: TranscriptResult(
            segments=[SimpleNamespace(start=0.0, end=0.0, text="chunk text", speaker=None)],
            text="chunk text",
        ),
    )

    from whisper_smith.transcribe import _transcribe_in_chunks

    result = _transcribe_in_chunks(audio_path, openai_client=object(), model="test-model")

    assert len(result.segments) == 2
    assert result.segments[0].start == 0.0
    assert result.segments[0].end == 59.4
    assert result.segments[1].start == 60.0
    assert result.segments[1].end == 94.2
