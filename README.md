# whisper-smith

[![PyPI version](https://img.shields.io/pypi/v/whisper-smith)](https://pypi.org/project/whisper-smith/)
[![Python versions](https://img.shields.io/pypi/pyversions/whisper-smith)](https://pypi.org/project/whisper-smith/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/yeiichi/whisper-smith/blob/main/LICENSE)
[![Docs](https://readthedocs.org/projects/whisper-smith/badge/?version=latest)](https://whisper-smith.readthedocs.io/)

`whisper-smith` is a small Python CLI/app helper for transcribing audio files with OpenAI speech-to-text models.

## Features

- Transcribe local audio files
- CLI-first workflow for quick terminal use
- Output as `txt`, `json`, `srt`, or `vtt`
- Automatically infer output format from output file extension
- Load environment variables from `.env`

## Run on Google Colab (free GPU)

No local setup needed. Open the notebook directly in Colab and run the full
speaker-aligned transcript pipeline on a free T4 GPU:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yeiichi/whisper-smith/blob/main/notebooks/colab_aligned_transcript.ipynb)

The notebook covers: install → set API keys → upload audio → run pipeline → download result.
Use a GPU runtime for the best diarization performance. The notebook also
includes an advanced GPU pipeline for explicitly moving the pyannote model to
CUDA.

## Requirements

- Python `3.10+`
- An OpenAI API key (`OPENAI_API_KEY`)
- For large-file fallback: either system `ffmpeg` in `PATH`, or Python package `imageio-ffmpeg`
- For optional speaker diarization: a Hugging Face token (`HUGGINGFACE_TOKEN`) and `pyannote.audio`

## Installation

### Option 1: uv (recommended)

```bash
uv sync
```

### Option 2: pip

```bash
pip install -e .
```

### Optional speaker diarization dependencies

```bash
uv sync --extra diarize
```

or:

```bash
pip install -e ".[diarize]"
```

## Configuration

Set your API key in the environment or in a `.env` file:

```bash
export OPENAI_API_KEY="your_api_key_here"
export HUGGINGFACE_TOKEN="your_huggingface_token_here"
```

Or create `.env` in project root:

```env
OPENAI_API_KEY=your_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here
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

### 6) Run speaker diarization

```bash
whisper-smith data/sample.m4a --diarize --output data/sample.diarization.json
```

Diarization currently supports JSON output only. Optional speaker hints:

```bash
whisper-smith data/sample.m4a --diarize --format json --num-speakers 2
```

### 7) Create speaker-aligned transcript JSON

Run the full pipeline from one audio file:

```bash
whisper-smith data/sample.m4a --align --output data/sample.aligned.json
```

This writes the main aligned transcript JSON to `data/sample.aligned.json` and
also writes intermediate artifacts beside it:

```text
data/sample.transcript.json
data/sample.diarization.json
```

To put the intermediate artifacts in a separate directory:

```bash
whisper-smith data/sample.m4a --align --output data/sample.aligned.json --artifacts-dir data/artifacts
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

### Speaker diarization

```python
from pathlib import Path
from whisper_smith.diarize import diarize_audio

result = diarize_audio(Path("data/sample.m4a"))

for segment in result.segments:
    print(segment.start, segment.end, segment.speaker)
```

`diarize_audio` uses `HUGGINGFACE_TOKEN` from the environment, or accepts
`hf_token="..."` explicitly.

The default local model is `pyannote/speaker-diarization-3.1`, which is compatible
with the Intel macOS dependency set. You may pass a different model explicitly
from Python or with `--diarization-model` when running on a newer platform.

## Notes

- If `--output` is omitted, transcript is printed to stdout.
- If `--format` is omitted, format is inferred from `--output` extension when possible.
- If an output file already exists, add `--overwrite` to replace it.
- Transcription uses a timestamp-capable OpenAI model by default so JSON, SRT,
  and VTT outputs have segment timestamps.
- For large audio files, `whisper-smith` automatically splits audio into chunks and
  merges transcript text.
- If diarization fails with `torchaudio` missing `AudioMetaData`, refresh the
  optional diarization dependencies with `uv lock --upgrade-package torch
  --upgrade-package torchaudio` and then `uv sync --extra diarize`.

## Development

Run tests:

```bash
pytest
```
