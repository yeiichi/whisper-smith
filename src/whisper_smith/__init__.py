# src/whisper_smith/__init__.py

__version__ = "0.1.0"

from .transcribe import transcribe_file

__all__ = [
    "transcribe_file",
]