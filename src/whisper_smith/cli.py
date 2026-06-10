import argparse
import sys
from pathlib import Path
from collections.abc import Sequence

from dotenv import load_dotenv

from whisper_smith.align import assign_speakers
from whisper_smith.diarize import DEFAULT_DIARIZATION_MODEL, diarize_audio
from whisper_smith.exporters import (
    export_diarization,
    export_diarization_json,
    export_json,
    export_transcript,
)
from whisper_smith.json_to_csv import main as json_to_csv_main
from whisper_smith.transcribe import transcribe_audio


SUPPORTED_OUTPUT_FORMATS = ("txt", "json", "srt", "vtt")
SUPPORTED_TIMESTAMP_FORMATS = ("seconds", "hms")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="whisper-smith",
        description="Transcribe audio files with OpenAI transcription models.",
        epilog="Subcommands: json-to-csv",
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
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Run speaker diarization instead of transcription. Only JSON output is supported.",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help=(
            "Run transcription and diarization, then write speaker-aligned JSON. "
            "Intermediate transcript and diarization JSON files are written by default "
            "when --output is used."
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        help=(
            "Directory for --align intermediate transcript and diarization JSON files. "
            "Defaults to the aligned output file directory."
        ),
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Do not write intermediate JSON files when using --align.",
    )
    parser.add_argument(
        "--diarization-model",
        help=(
            "Hugging Face pyannote pipeline to use for --diarize or --align. "
            f"Defaults to {DEFAULT_DIARIZATION_MODEL}."
        ),
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        help="Exact number of speakers to use for diarization.",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        help="Minimum number of speakers to use for diarization.",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        help="Maximum number of speakers to use for diarization.",
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


def resolve_alignment_artifact_paths(
    audio_path: Path,
    aligned_output_path: Path,
    artifacts_dir: Path | None,
) -> tuple[Path, Path]:
    artifact_parent = (
        artifacts_dir if artifacts_dir is not None else aligned_output_path.parent
    )
    return (
        artifact_parent / f"{audio_path.stem}.transcript.json",
        artifact_parent / f"{audio_path.stem}.diarization.json",
    )


def validate_mode_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    diarization_options = (
        args.diarization_model,
        args.num_speakers,
        args.min_speakers,
        args.max_speakers,
    )
    if args.diarize and args.align:
        parser.error("--diarize cannot be combined with --align.")

    if not (args.diarize or args.align) and any(
        option is not None for option in diarization_options
    ):
        parser.error("Diarization options require --diarize or --align.")

    if args.artifacts_dir is not None and not args.align:
        parser.error("--artifacts-dir requires --align.")

    if args.artifacts_dir is not None and args.output is None:
        parser.error("--artifacts-dir requires --output.")

    if args.no_artifacts and not args.align:
        parser.error("--no-artifacts requires --align.")

    if not (args.diarize or args.align):
        return

    if args.num_speakers is not None and (
        args.min_speakers is not None or args.max_speakers is not None
    ):
        parser.error(
            "--num-speakers cannot be combined with --min-speakers or --max-speakers"
        )

    if (
        args.min_speakers is not None
        and args.max_speakers is not None
        and args.min_speakers > args.max_speakers
    ):
        parser.error("--min-speakers cannot be greater than --max-speakers")


def run_diarization(args: argparse.Namespace, output_format: str) -> str:
    diarization = diarize_audio(
        args.audio_path,
        model=args.diarization_model or DEFAULT_DIARIZATION_MODEL,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )
    return export_diarization(diarization, output_format)


def run_transcription(args: argparse.Namespace, output_format: str) -> str:
    transcript = transcribe_audio(args.audio_path)
    return export_transcript(
        transcript,
        output_format,
        timestamp_format=args.timestamp_format,
    )


def run_alignment(args: argparse.Namespace) -> tuple[str, str, str]:
    transcript = transcribe_audio(args.audio_path)
    diarization = diarize_audio(
        args.audio_path,
        model=args.diarization_model or DEFAULT_DIARIZATION_MODEL,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )
    aligned = assign_speakers(transcript, diarization)
    return (
        export_json(aligned, timestamp_format=args.timestamp_format),
        export_json(transcript, timestamp_format=args.timestamp_format),
        export_diarization_json(diarization),
    )


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()

    argv_items = list(sys.argv[1:] if argv is None else argv)
    if argv_items[:1] == ["json-to-csv"]:
        json_to_csv_main(argv_items[1:])
        return

    parser = build_parser()
    args = parser.parse_args(argv_items)

    if not args.audio_path.is_file():
        parser.error(f"Audio file not found: {args.audio_path}")

    raw_output_arg = extract_raw_output_arg(argv_items)
    output_format = infer_output_format(args.output, args.format)
    validate_mode_args(args, parser)
    if args.diarize or args.align:
        output_is_directory_hint = args.output is not None and (
            args.output.is_dir()
            or (raw_output_arg is not None and raw_output_arg.endswith("/"))
        )
        if args.format is None and (args.output is None or output_is_directory_hint):
            output_format = "json"
        if output_format != "json":
            if args.align:
                parser.error("Aligned transcript output currently supports only JSON.")
            parser.error("Diarization output currently supports only JSON.")

    if args.output is None:
        if args.align:
            rendered_output, _transcript_json, _diarization_json = run_alignment(args)
        else:
            rendered_output = (
                run_diarization(args, output_format)
                if args.diarize
                else run_transcription(args, output_format)
            )
        print(rendered_output, end="")
        return

    resolved_output = resolve_output_path(
        audio_path=args.audio_path,
        output_path=args.output,
        output_format="aligned.json" if args.align else output_format,
        raw_output_arg=raw_output_arg,
    )

    try:
        ensure_output_path_is_writable(resolved_output, args.overwrite)
        if args.align and not args.no_artifacts:
            transcript_output, diarization_output = resolve_alignment_artifact_paths(
                args.audio_path,
                resolved_output,
                args.artifacts_dir,
            )
            ensure_output_path_is_writable(transcript_output, args.overwrite)
            ensure_output_path_is_writable(diarization_output, args.overwrite)
    except FileExistsError as error:
        parser.error(str(error))

    if args.align:
        rendered_output, transcript_json, diarization_json = run_alignment(args)
    else:
        rendered_output = (
            run_diarization(args, output_format)
            if args.diarize
            else run_transcription(args, output_format)
        )

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(rendered_output, encoding="utf-8")

    if args.align and not args.no_artifacts:
        transcript_output, diarization_output = resolve_alignment_artifact_paths(
            args.audio_path,
            resolved_output,
            args.artifacts_dir,
        )
        transcript_output.parent.mkdir(parents=True, exist_ok=True)
        transcript_output.write_text(transcript_json, encoding="utf-8")
        diarization_output.parent.mkdir(parents=True, exist_ok=True)
        diarization_output.write_text(diarization_json, encoding="utf-8")


if __name__ == "__main__":
    main()
