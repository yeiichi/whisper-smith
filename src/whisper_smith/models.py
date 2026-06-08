from dataclasses import dataclass


@dataclass(slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None


@dataclass(slots=True)
class TranscriptResult:
    segments: list[TranscriptSegment]
    text: str = ""

    def __post_init__(self) -> None:
        if not self.text:
            self.text = "\n".join(segment.text for segment in self.segments)
