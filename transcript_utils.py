import re
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    RequestBlocked,
    VideoUnavailable
)

# -------------------------------------------------
# Basic content filtering
# -------------------------------------------------
ABUSIVE_WORDS = [
    "fuck", "shit", "bitch", "asshole", "bastard"
]


def clean_text(text: str) -> str:
    text = text.lower()

    for word in ABUSIVE_WORDS:
        text = re.sub(
            rf"\b{word}\b",
            "[censored]",
            text,
            flags=re.IGNORECASE
        )

    text = re.sub(r"\s+", " ", text)
    return text.strip()


# -------------------------------------------------
# Transcript Fetcher (Robust & Safe)
# -------------------------------------------------
def get_clean_transcript(video_id: str) -> str:
    """
    Fetches and cleans YouTube transcript.
    Handles disabled transcripts, temporary blocks,
    and unavailable videos gracefully.
    """

    try:
        # Attempt to fetch transcript (manual or auto)
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        try:
            # Prefer manually created English transcript
            transcript = transcript_list.find_manually_created_transcript(["en"])
        except NoTranscriptFound:
            # Fallback to auto-generated English transcript
            transcript = transcript_list.find_generated_transcript(["en"])

        transcript_text = " ".join(
            chunk["text"] for chunk in transcript.fetch()
        )

        if not transcript_text.strip():
            raise NoTranscriptFound(video_id, ["en"], transcript_list)

        return clean_text(transcript_text)

    except TranscriptsDisabled:
        raise RuntimeError(
            "This YouTube video does not have captions enabled."
        )

    except NoTranscriptFound:
        raise RuntimeError(
            "No English transcript is available for this video."
        )

    except RequestBlocked:
        raise RuntimeError(
            "YouTube temporarily blocked transcript access for this video. "
            "Please try again later or use a different video."
        )

    except VideoUnavailable:
        raise RuntimeError(
            "This video is unavailable or private."
        )

    except Exception:
        raise RuntimeError(
            "Failed to retrieve transcript due to an unexpected error."
        )
