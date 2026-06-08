import os
from pathlib import Path
from typing import Any, Protocol

from whisper_smith.models import DiarizationResult, DiarizationSegment

DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"


class PyannotePipeline(Protocol):
    def __call__(self, audio_path: str, **kwargs: Any) -> Any: ...


def _load_pyannote_pipeline_class() -> Any:
    try:
        from pyannote.audio import Pipeline
    except ImportError as error:
        raise RuntimeError(
            "Speaker diarization requires pyannote.audio. "
            "Install it with 'uv sync --extra diarize' or "
            "'pip install whisper-smith[diarize]'."
        ) from error

    return Pipeline


def _resolve_hf_token(hf_token: str | None) -> str | None:
    return (
        hf_token
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("PYANNOTE_AUTH_TOKEN")
    )


def from_pyannote_output(output: Any) -> DiarizationResult:
    diarization = (
        getattr(output, "exclusive_speaker_diarization", None)
        or getattr(output, "speaker_diarization", None)
        or output
    )

    if not hasattr(diarization, "itertracks"):
        raise TypeError("Unsupported pyannote diarization output.")

    segments: list[DiarizationSegment] = []
    for turn, _track, speaker in diarization.itertracks(yield_label=True):
        segments.append(
            DiarizationSegment(
                start=float(turn.start),
                end=float(turn.end),
                speaker=str(speaker),
            )
        )

    return DiarizationResult(segments=segments)


def diarize_audio(
    audio_path: str | Path,
    *,
    hf_token: str | None = None,
    model: str = DEFAULT_DIARIZATION_MODEL,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    pipeline: PyannotePipeline | None = None,
) -> DiarizationResult:
    path = Path(audio_path)

    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    diarization_pipeline = pipeline
    if diarization_pipeline is None:
        token = _resolve_hf_token(hf_token)
        if not token:
            raise RuntimeError(
                "Hugging Face token not found. Set HUGGINGFACE_TOKEN in your "
                "environment or .env file, or pass hf_token explicitly."
            )

        Pipeline = _load_pyannote_pipeline_class()
        diarization_pipeline = Pipeline.from_pretrained(model, token=token)

    pipeline_kwargs: dict[str, int] = {}
    if num_speakers is not None:
        pipeline_kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        pipeline_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        pipeline_kwargs["max_speakers"] = max_speakers

    output = diarization_pipeline(str(path), **pipeline_kwargs)
    return from_pyannote_output(output)


def diarize_file(
    audio_path: str | Path,
    *,
    hf_token: str | None = None,
    model: str = DEFAULT_DIARIZATION_MODEL,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> DiarizationResult:
    return diarize_audio(
        audio_path,
        hf_token=hf_token,
        model=model,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
