import builtins
import sys
import types
from types import SimpleNamespace

import pytest

from whisper_smith.diarize import (
    DEFAULT_DIARIZATION_MODEL,
    _allow_trusted_pyannote_checkpoint_globals,
    _load_pyannote_pipeline_class,
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


def test_load_pyannote_pipeline_reports_incompatible_torchaudio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyannote.audio":
            raise AttributeError("module 'torchaudio' has no attribute 'AudioMetaData'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="dependency versions are incompatible"):
        _load_pyannote_pipeline_class()


def test_allow_trusted_pyannote_checkpoint_globals_registers_checkpoint_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    import torch
    from torch.torch_version import TorchVersion

    class FakeSpecifications:
        pass

    class FakeProblem:
        pass

    class FakeResolution:
        pass

    class FakeScope:
        pass

    fake_task_module = types.ModuleType("pyannote.audio.core.task")
    fake_task_module.Problem = FakeProblem
    fake_task_module.Resolution = FakeResolution
    fake_task_module.Scope = FakeScope
    fake_task_module.Specifications = FakeSpecifications
    monkeypatch.setitem(sys.modules, "pyannote.audio.core.task", fake_task_module)

    monkeypatch.setattr(
        torch.serialization,
        "add_safe_globals",
        lambda globals_to_add: calls.append(globals_to_add),
        raising=False,
    )

    _allow_trusted_pyannote_checkpoint_globals()

    assert calls == [
        [TorchVersion, FakeSpecifications, FakeProblem, FakeResolution, FakeScope]
    ]


def test_diarize_audio_passes_speaker_options_to_pipeline(tmp_path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"dummy audio")
    converted_path = tmp_path / "converted.wav"
    converted_path.write_bytes(b"converted audio")
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

    from whisper_smith import diarize

    original_convert = diarize._convert_audio_for_diarization
    diarize._convert_audio_for_diarization = lambda *_args: converted_path
    try:
        result = diarize_audio(
            audio_path,
            num_speakers=2,
            min_speakers=1,
            max_speakers=3,
            pipeline=fake_pipeline,
        )
    finally:
        diarize._convert_audio_for_diarization = original_convert

    assert calls == [
        (
            str(converted_path),
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
    converted_path = tmp_path / "converted.wav"
    converted_path.write_bytes(b"converted audio")
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
    monkeypatch.setattr(
        "whisper_smith.diarize._convert_audio_for_diarization",
        lambda *_args: converted_path,
    )

    result = diarize_audio(audio_path)

    assert calls == [(DEFAULT_DIARIZATION_MODEL, "hf_test")]
    assert result.segments[0].speaker == "SPEAKER_00"


def test_diarize_audio_supports_pyannote_use_auth_token_keyword(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"dummy audio")
    converted_path = tmp_path / "converted.wav"
    converted_path.write_bytes(b"converted audio")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_test")
    calls = []

    class FakePipeline:
        @staticmethod
        def from_pretrained(model, **kwargs):
            calls.append((model, kwargs))
            if "token" in kwargs:
                raise TypeError("got an unexpected keyword argument 'token'")

            def fake_pipeline(path, **pipeline_kwargs):
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
    monkeypatch.setattr(
        "whisper_smith.diarize._convert_audio_for_diarization",
        lambda *_args: converted_path,
    )

    result = diarize_audio(audio_path)

    assert calls == [
        (DEFAULT_DIARIZATION_MODEL, {"token": "hf_test"}),
        (DEFAULT_DIARIZATION_MODEL, {"use_auth_token": "hf_test"}),
    ]
    assert result.segments[0].speaker == "SPEAKER_00"


def test_diarize_audio_raises_when_pyannote_returns_no_pipeline(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"dummy audio")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_test")

    class FakePipeline:
        @staticmethod
        def from_pretrained(*_args, **_kwargs):
            return None

    monkeypatch.setattr(
        "whisper_smith.diarize._load_pyannote_pipeline_class",
        lambda: FakePipeline,
    )

    with pytest.raises(RuntimeError, match="Could not load pyannote diarization model"):
        diarize_audio(audio_path)


def test_diarize_audio_raises_when_audio_conversion_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_path = tmp_path / "audio.mp4"
    audio_path.write_bytes(b"dummy audio")

    def fail_conversion(*_args):
        raise subprocess.CalledProcessError(1, ["ffmpeg"])

    import subprocess

    monkeypatch.setattr(
        "whisper_smith.diarize._convert_audio_for_diarization",
        fail_conversion,
    )

    with pytest.raises(RuntimeError, match="Failed to prepare audio"):
        diarize_audio(audio_path, pipeline=lambda *_args, **_kwargs: None)
