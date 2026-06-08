from whisper_smith.align import assign_speakers
from whisper_smith.models import (
    DiarizationResult,
    DiarizationSegment,
    TranscriptResult,
    TranscriptSegment,
)


def test_assign_speakers_uses_largest_overlap() -> None:
    transcript = TranscriptResult(
        segments=[
            TranscriptSegment(start=0.0, end=2.0, text="Hello."),
            TranscriptSegment(start=2.0, end=4.0, text="World."),
        ]
    )
    diarization = DiarizationResult(
        segments=[
            DiarizationSegment(start=0.0, end=0.5, speaker="SPEAKER_00"),
            DiarizationSegment(start=0.5, end=2.0, speaker="SPEAKER_01"),
            DiarizationSegment(start=2.0, end=4.0, speaker="SPEAKER_00"),
        ]
    )

    result = assign_speakers(transcript, diarization)

    assert [segment.speaker for segment in result.segments] == [
        "SPEAKER_01",
        "SPEAKER_00",
    ]


def test_assign_speakers_preserves_text_and_original_transcript() -> None:
    transcript = TranscriptResult(
        text="Custom text",
        segments=[
            TranscriptSegment(start=0.0, end=1.0, text="Hello."),
        ],
    )
    diarization = DiarizationResult(
        segments=[
            DiarizationSegment(start=0.0, end=1.0, speaker="SPEAKER_00"),
        ]
    )

    result = assign_speakers(transcript, diarization)

    assert result.text == "Custom text"
    assert result.segments[0].speaker == "SPEAKER_00"
    assert transcript.segments[0].speaker is None


def test_assign_speakers_keeps_existing_speaker_when_no_segment_overlaps() -> None:
    transcript = TranscriptResult(
        segments=[
            TranscriptSegment(
                start=0.0,
                end=1.0,
                text="Hello.",
                speaker="EXISTING",
            ),
        ]
    )
    diarization = DiarizationResult(
        segments=[
            DiarizationSegment(start=2.0, end=3.0, speaker="SPEAKER_00"),
        ]
    )

    result = assign_speakers(transcript, diarization)

    assert result.segments[0].speaker == "EXISTING"
