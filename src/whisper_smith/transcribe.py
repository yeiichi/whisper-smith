import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Protocol

from openai import BadRequestError, OpenAI

from whisper_smith.models import TranscriptResult, TranscriptSegment

MAX_SINGLE_UPLOAD_BYTES = 24 * 1024 * 1024
CHUNK_SECONDS = 60


class OpenAITranscriptionResponse(Protocol):
    text: str


def from_openai_response(response: OpenAITranscriptionResponse) -> TranscriptResult:
    response_segments = getattr(response, "segments", None)
    if isinstance(response_segments, list) and response_segments:
        segments: list[TranscriptSegment] = []
        for segment in response_segments:
            text = str(getattr(segment, "text", "")).strip()
            if not text:
                continue

            segments.append(
                TranscriptSegment(
                    start=float(getattr(segment, "start", 0.0)),
                    end=float(getattr(segment, "end", 0.0)),
                    text=text,
                    speaker=getattr(segment, "speaker", None),
                )
            )

        if segments:
            return TranscriptResult(
                segments=segments,
                text=str(getattr(response, "text", "")).strip(),
            )

    return TranscriptResult(
        segments=[
            TranscriptSegment(
                start=0.0,
                end=0.0,
                text=response.text.strip(),
            )
        ]
    )


def _transcribe_single_file(
    path: Path,
    openai_client: OpenAI,
    model: str,
) -> TranscriptResult:
    def _create_transcription_with_segments() -> OpenAITranscriptionResponse:
        with path.open("rb") as audio_file:
            return openai_client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

    def _create_transcription_plain_json() -> OpenAITranscriptionResponse:
        with path.open("rb") as audio_file:
            return openai_client.audio.transcriptions.create(
                model=model,
                file=audio_file,
                response_format="json",
            )

    try:
        response = _create_transcription_with_segments()
    except BadRequestError as error:
        if _is_response_format_unsupported(error):
            response = _create_transcription_plain_json()
        else:
            raise

    return from_openai_response(response)


def _split_audio_into_chunks(path: Path, output_dir: Path) -> list[Path]:
    # Use small fixed segments to avoid needing ffprobe. One minute chunks keep
    # request payloads comfortably below API file-size limits in most audio formats.
    chunk_seconds = CHUNK_SECONDS

    ffmpeg_executable = _resolve_ffmpeg_executable()
    output_pattern = output_dir / f"chunk_%05d{path.suffix}"
    command = [
        ffmpeg_executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "segment",
        "-segment_time",
        str(chunk_seconds),
        "-c",
        "copy",
        "-map",
        "0:a",
        str(output_pattern),
    ]
    subprocess.run(command, check=True)
    chunks = sorted(output_dir.glob(f"chunk_*{path.suffix}"))
    if not chunks:
        raise RuntimeError("Failed to split audio file into chunks.")
    return chunks


def _get_audio_duration_seconds(path: Path) -> float | None:
    ffprobe_executable = shutil.which("ffprobe")
    if not ffprobe_executable:
        return None

    command = [
        ffprobe_executable,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None

    try:
        duration = float(result.stdout.strip())
    except ValueError:
        return None

    if duration <= 0:
        return None
    return duration


def _resolve_ffmpeg_executable() -> str:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path

    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError(
            "Chunked transcription requires ffmpeg. Install system ffmpeg or add the "
            "'imageio-ffmpeg' package."
        ) from error

    return imageio_ffmpeg.get_ffmpeg_exe()


def _transcribe_in_chunks(
    path: Path,
    openai_client: OpenAI,
    model: str,
) -> TranscriptResult:
    merged_segments: list[TranscriptSegment] = []
    collected_texts: list[str] = []

    try:
        with tempfile.TemporaryDirectory(prefix="whisper_smith_chunks_") as temp_dir:
            chunks = _split_audio_into_chunks(path, Path(temp_dir))
            for index, chunk in enumerate(chunks):
                chunk_result = _transcribe_single_file(chunk, openai_client, model)
                chunk_offset_seconds = float(index * CHUNK_SECONDS)
                chunk_duration = _get_audio_duration_seconds(chunk) or float(CHUNK_SECONDS)
                for segment in chunk_result.segments:
                    shifted_start = segment.start + chunk_offset_seconds
                    shifted_end = segment.end + chunk_offset_seconds
                    if shifted_end <= shifted_start:
                        shifted_end = chunk_offset_seconds + chunk_duration
                    merged_segments.append(
                        TranscriptSegment(
                            start=shifted_start,
                            end=shifted_end,
                            text=segment.text,
                            speaker=segment.speaker,
                        )
                    )
                if chunk_result.text:
                    collected_texts.append(chunk_result.text)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "Failed to split audio for chunked transcription. The file may be malformed."
        ) from error

    return TranscriptResult(
        segments=merged_segments
        or [
            TranscriptSegment(
                start=0.0,
                end=0.0,
                text="\n".join(collected_texts).strip(),
            )
        ],
        text="\n".join(collected_texts).strip(),
    )


def _is_probably_oversize_or_container_error(error: BadRequestError) -> bool:
    if not isinstance(error.body, dict):
        return False
    payload = error.body.get("error")
    if not isinstance(payload, dict):
        return False

    message = str(payload.get("message", "")).lower()
    param = payload.get("param")
    code = payload.get("code")
    return param == "file" and code == "invalid_value" and (
        "unsupported" in message or "corrupted" in message
    )


def _is_response_format_unsupported(error: BadRequestError) -> bool:
    error_text = str(error).lower()

    if isinstance(error.body, dict):
        payload = error.body.get("error")
        if isinstance(payload, dict):
            param = payload.get("param")
            code = payload.get("code")
            message = str(payload.get("message", "")).lower()
            if (
                param == "response_format"
                and code == "unsupported_value"
                and "response_format" in message
            ):
                return True

    return (
        "response_format" in error_text
        and "verbose_json" in error_text
        and "not compatible" in error_text
    )


def transcribe_audio(
    audio_path: str | Path,
    *,
    client: OpenAI | None = None,
    model: str = "gpt-4o-transcribe",
) -> TranscriptResult:
    path = Path(audio_path)

    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    openai_client = client or OpenAI()

    if path.stat().st_size > MAX_SINGLE_UPLOAD_BYTES:
        return _transcribe_in_chunks(path, openai_client, model)

    try:
        return _transcribe_single_file(path, openai_client, model)
    except BadRequestError as error:
        if _is_probably_oversize_or_container_error(error):
            return _transcribe_in_chunks(path, openai_client, model)
        raise


def transcribe_file(
    audio_path: str | Path,
    *,
    client: OpenAI | None = None,
    model: str = "gpt-4o-transcribe",
) -> TranscriptResult:
    return transcribe_audio(audio_path, client=client, model=model)
