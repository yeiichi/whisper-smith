from types import SimpleNamespace

import pytest

from whisper_smith.diarize import (
    DEFAULT_DIARIZATION_MODEL,
    diarize_audio,
    from_pyannote_output,
)
from whisper_smith.models import DiarizationResult


class FakeAnnotation:
    def __init__(self, rows):
        self.rows = rows

    def itertracks(self, yield_label: bool = False):
        assert yield_label is True
        yield from self.rows


def test_from_pyannote_output_uses_exclusive_diarization_when_available() -> None:
    regular = FakeAnnotation(
        [
            (
                SimpleNamespace(start=0.0, end=1.0),
                "track",
                "REGULAR",
            )
        ]
    )
    exclusive = FakeAnnotation(
        [
            (
                SimpleNamespace(start=1.5, end=3.0),
                "track",
                "SPEAKER_00",
            )
        ]
    )

    result = from_pyannote_output(
        SimpleNamespace(
            speaker_diarization=regular,
            exclusive_speaker_diarization=exclusive,
        )
    )

    assert isinstance(result, DiarizationResult)
    assert len(result.segments) == 1
    assert result.segments[0].start == 1.5
    assert result.segments[0].end == 3.0
    assert result.segments[0].speaker == "SPEAKER_00"


def test_diarize_audio_raises_for_missing_file() -> None:
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        diarize_audio(
            "does-not-exist.mp3",
            pipeline=lambda *_args, **_kwargs: None,
        )


def test_diarize_audio_requires_token_without_pipeline(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"dummy audio")
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.delenv("PYANNOTE_AUTH_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="Hugging Face token not found"):
        diarize_audio(audio_path)


def test_diarize_audio_passes_speaker_options_to_pipeline(tmp_path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"dummy audio")
    calls = []

    def fake_pipeline(path, **kwargs):
        calls.append((path, kwargs))
        return FakeAnnotation(
            [
                (
                    SimpleNamespace(start=0.25, end=2.5),
                    "track",
                    "SPEAKER_01",
                )
            ]
        )

    result = diarize_audio(
        audio_path,
        num_speakers=2,
        min_speakers=1,
        max_speakers=3,
        pipeline=fake_pipeline,
    )

    assert calls == [
        (
            str(audio_path),
            {
                "num_speakers": 2,
                "min_speakers": 1,
                "max_speakers": 3,
            },
        )
    ]
    assert result.segments[0].speaker == "SPEAKER_01"


def test_diarize_audio_loads_pyannote_pipeline_with_env_token(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"dummy audio")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_test")
    calls = []

    class FakePipeline:
        @staticmethod
        def from_pretrained(model, token):
            calls.append((model, token))

            def fake_pipeline(path, **kwargs):
                return FakeAnnotation(
                    [
                        (
                            SimpleNamespace(start=0.0, end=1.0),
                            "track",
                            "SPEAKER_00",
                        )
                    ]
                )

            return fake_pipeline

    monkeypatch.setattr(
        "whisper_smith.diarize._load_pyannote_pipeline_class",
        lambda: FakePipeline,
    )

    result = diarize_audio(audio_path)

    assert calls == [(DEFAULT_DIARIZATION_MODEL, "hf_test")]
    assert result.segments[0].speaker == "SPEAKER_00"
