import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Protocol

from whisper_smith.models import DiarizationResult, DiarizationSegment

DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"


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


def _resolve_ffmpeg_executable() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError(
            "Speaker diarization requires ffmpeg for media conversion. "
            "Install system ffmpeg or add the 'imageio-ffmpeg' package."
        ) from error

    return imageio_ffmpeg.get_ffmpeg_exe()


def _convert_audio_for_diarization(path: Path, output_dir: Path) -> Path:
    wav_path = output_dir / f"{path.stem}.diarization.wav"
    command = [
        _resolve_ffmpeg_executable(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-y",
        str(wav_path),
    ]
    subprocess.run(command, check=True)
    return wav_path


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
        try:
            diarization_pipeline = Pipeline.from_pretrained(model, token=token)
        except TypeError as error:
            if "token" not in str(error):
                raise
            diarization_pipeline = Pipeline.from_pretrained(
                model,
                use_auth_token=token,
            )

        if diarization_pipeline is None:
            raise RuntimeError(
                f"Could not load pyannote diarization model: {model}. "
                "Check that HUGGINGFACE_TOKEN is valid and that you accepted the "
                "model's Hugging Face user conditions."
            )

    pipeline_kwargs: dict[str, int] = {}
    if num_speakers is not None:
        pipeline_kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        pipeline_kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        pipeline_kwargs["max_speakers"] = max_speakers

    try:
        with tempfile.TemporaryDirectory(prefix="whisper_smith_diarize_") as temp_dir:
            diarization_path = _convert_audio_for_diarization(path, Path(temp_dir))
            output = diarization_pipeline(str(diarization_path), **pipeline_kwargs)
            return from_pyannote_output(output)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "Failed to prepare audio for speaker diarization. The file may be malformed."
        ) from error


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
