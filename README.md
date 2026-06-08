# whisper-smith

`whisper-smith` is a small Python CLI/app helper for transcribing audio files with OpenAI speech-to-text models.

## Features

- Transcribe local audio files
- CLI-first workflow for quick terminal use
- Output as `txt`, `json`, `srt`, or `vtt`
- Automatically infer output format from output file extension
- Load environment variables from `.env`

## Requirements

- Python `3.10+`
- An OpenAI API key (`OPENAI_API_KEY`)
- For large-file fallback: either system `ffmpeg` in `PATH`, or Python package `imageio-ffmpeg`

## Installation

### Option 1: uv (recommended)

```bash
uv sync
```

### Option 2: pip

```bash
pip install -e .
```

## Configuration

Set your API key in the environment or in a `.env` file:

```bash
export OPENAI_API_KEY="your_api_key_here"
```

Or create `.env` in project root:

```env
OPENAI_API_KEY=your_api_key_here
```

## CLI Usage Guide

Basic command:

```bash
whisper-smith <audio_path>
```

Show help:

```bash
whisper-smith --help
```

### 1) Print transcript to terminal (default `txt`)

```bash
whisper-smith data/sample.m4a
```

### 2) Save transcript to a file

```bash
whisper-smith data/sample.m4a --output data/sample.txt
```

### 3) Choose output format explicitly

```bash
whisper-smith data/sample.m4a --format json --output data/sample.json
```

Supported CLI formats: `txt`, `json`, `srt`, `vtt`

### 4) Let format be inferred from output extension

```bash
whisper-smith data/sample.m4a --output data/sample.srt
```

### 5) Overwrite existing file

```bash
whisper-smith data/sample.m4a --output data/sample.txt --overwrite
```

## Python Usage

```python
from pathlib import Path
from whisper_smith.transcribe import transcribe_audio
from whisper_smith.exporters import export_transcript

result = transcribe_audio(Path("data/sample.m4a"))
print(result.text)

srt = export_transcript(result, "srt")
Path("data/sample.srt").write_text(srt, encoding="utf-8")
```

## Notes

- If `--output` is omitted, transcript is printed to stdout.
- If `--format` is omitted, format is inferred from `--output` extension when possible.
- If an output file already exists, add `--overwrite` to replace it.
- For large audio files, `whisper-smith` automatically splits audio into chunks and
  merges transcript text.

## Development

Run tests:

```bash
pytest
```
