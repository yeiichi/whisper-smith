import argparse
import sys
from pathlib import Path
from collections.abc import Sequence

from dotenv import load_dotenv

from whisper_smith.exporters import export_transcript
from whisper_smith.transcribe import transcribe_audio


SUPPORTED_OUTPUT_FORMATS = ("txt", "json", "srt", "vtt")
SUPPORTED_TIMESTAMP_FORMATS = ("seconds", "hms")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="whisper-smith",
        description="Transcribe audio files with OpenAI transcription models.",
    )
    parser.add_argument(
        "audio_path",
        type=Path,
        help="Path to the audio file to transcribe.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Path to write the transcript output. "
            "If omitted, the rendered transcript is printed to stdout."
        ),
    )
    parser.add_argument(
        "-f",
        "--format",
        choices=SUPPORTED_OUTPUT_FORMATS,
        default=None,
        help=(
            "Output format. Supported values: txt, json, srt, vtt. "
            "If omitted, the format is inferred from --output suffix, "
            "falling back to txt."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--timestamp-format",
        choices=SUPPORTED_TIMESTAMP_FORMATS,
        default="seconds",
        help=(
            "Timestamp format for JSON output: "
            "'seconds' (default) or 'hms' (HH:MM:SS). "
            "Ignored for txt/srt/vtt."
        ),
    )
    return parser


def infer_output_format(output_path: Path | None, requested_format: str | None) -> str:
    if requested_format is not None:
        return requested_format

    if output_path is None:
        return "txt"

    suffix = output_path.suffix.removeprefix(".").lower()
    if suffix in SUPPORTED_OUTPUT_FORMATS:
        return suffix

    return "txt"


def resolve_output_path(
    audio_path: Path,
    output_path: Path,
    output_format: str,
    raw_output_arg: str | None = None,
) -> Path:
    is_directory_hint = output_path.is_dir() or (
        raw_output_arg is not None and raw_output_arg.endswith("/")
    )

    if is_directory_hint:
        return output_path / f"{audio_path.stem}.{output_format}"

    return output_path


def extract_raw_output_arg(argv: Sequence[str] | None) -> str | None:
    argv_items = list(sys.argv[1:] if argv is None else argv)

    for index, token in enumerate(argv_items):
        if token in {"-o", "--output"} and index + 1 < len(argv_items):
            return argv_items[index + 1]
        if token.startswith("--output="):
            return token.split("=", 1)[1]
        if token.startswith("-o") and token != "-o":
            return token[2:]

    return None


def ensure_output_path_is_writable(output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. "
            "Use --overwrite to replace it."
        )


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.audio_path.is_file():
        parser.error(f"Audio file not found: {args.audio_path}")

    output_format = infer_output_format(args.output, args.format)
    if args.output is None:
        transcript = transcribe_audio(args.audio_path)
        rendered_transcript = export_transcript(
            transcript,
            output_format,
            timestamp_format=args.timestamp_format,
        )
        print(rendered_transcript, end="")
        return

    resolved_output = resolve_output_path(
        audio_path=args.audio_path,
        output_path=args.output,
        output_format=output_format,
        raw_output_arg=extract_raw_output_arg(argv),
    )

    try:
        ensure_output_path_is_writable(resolved_output, args.overwrite)
    except FileExistsError as error:
        parser.error(str(error))

    transcript = transcribe_audio(args.audio_path)
    rendered_transcript = export_transcript(
        transcript,
        output_format,
        timestamp_format=args.timestamp_format,
    )

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(rendered_transcript, encoding="utf-8")


if __name__ == "__main__":
    main()
