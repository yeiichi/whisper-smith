from whisper_smith.models import (
    DiarizationResult,
    DiarizationSegment,
    TranscriptResult,
    TranscriptSegment,
)


def assign_speakers(
    transcript: TranscriptResult,
    diarization: DiarizationResult,
) -> TranscriptResult:
    segments = [
        TranscriptSegment(
            start=segment.start,
            end=segment.end,
            text=segment.text,
            speaker=_best_speaker_for_segment(segment, diarization.segments),
        )
        for segment in transcript.segments
    ]
    return TranscriptResult(segments=segments, text=transcript.text)


def _best_speaker_for_segment(
    segment: TranscriptSegment,
    diarization_segments: list[DiarizationSegment],
) -> str | None:
    best_speaker = segment.speaker
    best_overlap = 0.0

    for diarization_segment in diarization_segments:
        overlap = _overlap_duration(segment, diarization_segment)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = diarization_segment.speaker

    return best_speaker


def _overlap_duration(
    transcript_segment: TranscriptSegment,
    diarization_segment: DiarizationSegment,
) -> float:
    start = max(transcript_segment.start, diarization_segment.start)
    end = min(transcript_segment.end, diarization_segment.end)
    return max(0.0, end - start)
