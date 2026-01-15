import re
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

ABUSIVE_WORDS = [
    "fuck", "shit", "bitch", "asshole", "bastard"
]

def clean_text(text: str) -> str:
    text = text.lower()
    for word in ABUSIVE_WORDS:
        text = re.sub(rf"\b{word}\b", "[censored]", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def get_clean_transcript(video_id: str) -> str:
    try:
        api = YouTubeTranscriptApi()
        transcript_list = api.fetch(video_id, languages=["en"])

        # ✅ FIX: access `.text` attribute
        transcript = " ".join(chunk.text for chunk in transcript_list)

        return clean_text(transcript)

    except TranscriptsDisabled:
        raise RuntimeError("Transcript not available for this video")
