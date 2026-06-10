import json
from pathlib import Path

import pytest

from whisper_smith.json_to_csv import (
    CSV_COLUMNS,
    load_segments,
    main as json_to_csv_main,
    render_csv,
)
from whisper_smith.cli import main as cli_main


def test_render_csv_includes_human_datetime_column() -> None:
    output = render_csv(
        [
            {
                "start": 1.5,
                "end": 3.0,
                "speaker": "SPEAKER_00",
                "text": "Hello, world.",
            }
        ],
        initial_datetime="2026-06-10T09:00:00",
    )

    assert output == (
        "start,end,start_dttm,speaker,text\n"
        '1.5,3.0,2026-06-10 09:00:01.500000,SPEAKER_00,"Hello, world."\n'
    )


def test_render_csv_accepts_hms_timestamps() -> None:
    output = render_csv(
        [
            {
                "start": "00:01:30",
                "end": "00:01:35",
                "speaker": None,
                "text": "Hi.",
            }
        ],
        initial_datetime="2026-06-10T09:00:00",
    )

    assert output == (
        "start,end,start_dttm,speaker,text\n"
        "00:01:30,00:01:35,2026-06-10 09:01:30,,Hi.\n"
    )


def test_load_segments_requires_segments_list(tmp_path: Path) -> None:
    json_file = tmp_path / "bad.json"
    json_file.write_text(json.dumps({"text": "missing segments"}), encoding="utf-8")

    with pytest.raises(ValueError, match="segments"):
        load_segments(json_file)


def test_csv_columns_match_public_shape() -> None:
    assert CSV_COLUMNS == ("start", "end", "start_dttm", "speaker", "text")


def test_cli_writes_csv_file(tmp_path: Path) -> None:
    json_file = tmp_path / "sample.aligned.json"
    csv_file = tmp_path / "sample.csv"
    json_file.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "speaker": "SPEAKER_00",
                        "text": "Hello.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    json_to_csv_main([str(json_file), "--output", str(csv_file)])

    assert csv_file.read_text(encoding="utf-8") == (
        "start,end,start_dttm,speaker,text\n"
        "0.0,1.0,1970-01-01 00:00:00,SPEAKER_00,Hello.\n"
    )


def test_cli_refuses_to_overwrite_existing_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    json_file = tmp_path / "sample.aligned.json"
    csv_file = tmp_path / "sample.csv"
    json_file.write_text(json.dumps({"segments": []}), encoding="utf-8")
    csv_file.write_text("existing", encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        json_to_csv_main([str(json_file), "--output", str(csv_file)])

    assert exc_info.value.code == 2
    assert csv_file.read_text(encoding="utf-8") == "existing"
    captured = capsys.readouterr()
    assert "Output file already exists" in captured.err


def test_cli_dispatches_json_to_csv_subcommand(tmp_path: Path) -> None:
    json_file = tmp_path / "sample.aligned.json"
    csv_file = tmp_path / "sample.csv"
    json_file.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "speaker": "SPEAKER_00",
                        "text": "Hello.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    cli_main(["json-to-csv", str(json_file), "--output", str(csv_file)])

    assert csv_file.read_text(encoding="utf-8") == (
        "start,end,start_dttm,speaker,text\n"
        "0.0,1.0,1970-01-01 00:00:00,SPEAKER_00,Hello.\n"
    )
