from __future__ import annotations

import argparse
import csv
import io
import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TextIO


CSV_COLUMNS = ("start", "end", "start_dttm", "speaker", "text")
DEFAULT_INITIAL_DATETIME = "1970-01-01T00:00:00"


def load_segments(json_path: Path) -> list[Mapping[str, Any]]:
    with json_path.open(encoding="utf-8") as json_file:
        payload = json.load(json_file)

    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object with a 'segments' list.")

    segments = payload.get("segments")
    if not isinstance(segments, list):
        raise ValueError("Expected JSON object with a 'segments' list.")

    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("Expected every segment to be a JSON object.")

    return segments


def render_csv(
    segments: Iterable[Mapping[str, Any]],
    initial_datetime: str = DEFAULT_INITIAL_DATETIME,
) -> str:
    output = io.StringIO()
    write_csv(segments, output, initial_datetime=initial_datetime)
    return output.getvalue()


def write_csv(
    segments: Iterable[Mapping[str, Any]],
    output_file: TextIO,
    initial_datetime: str = DEFAULT_INITIAL_DATETIME,
) -> None:
    initial_dttm = datetime.fromisoformat(initial_datetime)
    writer = csv.DictWriter(output_file, fieldnames=CSV_COLUMNS, lineterminator="\n")
    writer.writeheader()

    for segment in segments:
        start = segment.get("start", "")
        start_dttm = ""
        if start != "":
            start_dttm = (
                initial_dttm + timedelta(seconds=_timestamp_to_seconds(start))
            ).isoformat(sep=" ")

        writer.writerow(
            {
                "start": start,
                "end": segment.get("end", ""),
                "start_dttm": start_dttm,
                "speaker": segment.get("speaker") or "",
                "text": segment.get("text", ""),
            }
        )


def build_parser(prog: str = "whisper-smith json-to-csv") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Convert whisper-smith JSON output to human-friendly CSV.",
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Path to a whisper-smith transcript or aligned JSON file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to write CSV output. If omitted, CSV is printed to stdout.",
    )
    parser.add_argument(
        "--initial-datetime",
        default=DEFAULT_INITIAL_DATETIME,
        help=(
            "Datetime to add to each segment start for the start_dttm column. "
            f"Defaults to {DEFAULT_INITIAL_DATETIME}."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.json_path.is_file():
        parser.error(f"JSON file not found: {args.json_path}")

    try:
        segments = load_segments(args.json_path)
        rendered_csv = render_csv(segments, initial_datetime=args.initial_datetime)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    if args.output is None:
        print(rendered_csv, end="")
        return

    if args.output.exists() and not args.overwrite:
        parser.error(
            f"Output file already exists: {args.output}. "
            "Use --overwrite to replace it."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered_csv, encoding="utf-8")


def _timestamp_to_seconds(value: Any) -> float:
    if isinstance(value, int | float):
        return float(value)

    if not isinstance(value, str):
        raise ValueError(f"Unsupported timestamp value: {value!r}")

    try:
        return float(value)
    except ValueError:
        pass

    parts = value.split(":")
    if len(parts) != 3:
        raise ValueError(f"Unsupported timestamp value: {value!r}")

    hours, minutes, seconds = parts
    return (int(hours) * 3600) + (int(minutes) * 60) + float(seconds)


if __name__ == "__main__":
    main()
