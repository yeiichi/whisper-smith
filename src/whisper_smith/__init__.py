# src/whisper_smith/__init__.py

__version__ = "0.1.0"

from .diarize import diarize_file
from .transcribe import transcribe_file

__all__ = [
    "diarize_file",
    "transcribe_file",
]
