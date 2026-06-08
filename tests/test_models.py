from whisper_smith.models import TranscriptResult, TranscriptSegment


def test_transcript_segment_allows_optional_speaker() -> None:
    segment = TranscriptSegment(
        start=0.0,
        end=1.5,
        text="Hello.",
    )

    assert segment.speaker is None


def test_transcript_text_joins_segments() -> None:
    transcript = TranscriptResult(
        segments=[
            TranscriptSegment(start=0.0, end=1.0, text="Hello."),
            TranscriptSegment(start=1.0, end=2.0, text="World."),
        ]
    )

    assert transcript.text == "Hello.\nWorld."
