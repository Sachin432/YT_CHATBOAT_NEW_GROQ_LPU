import streamlit as st
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    RequestBlocked
)

from transcript_utils import get_clean_transcript
from rag_pipeline import build_chain


# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    layout="wide"
)

st.title("YouTube Video Chatbot (GROQ_MODEL: llama-3.1-8b-instant)")
st.caption("Ask questions directly from a YouTube video using AI")


# -------------------------------------------------
# Helper: Extract video ID
# -------------------------------------------------
def extract_video_id(url: str) -> str | None:
    try:
        parsed = urlparse(url)

        if "youtube.com" in parsed.netloc:
            return parse_qs(parsed.query).get("v", [None])[0]

        if "youtu.be" in parsed.netloc:
            return parsed.path.lstrip("/")

    except Exception:
        return None

    return None


# -------------------------------------------------
# Session state init
# -------------------------------------------------
if "chain" not in st.session_state:
    st.session_state.chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -------------------------------------------------
# Layout
# -------------------------------------------------
left, right = st.columns([1.2, 2])


# -------------------------------------------------
# LEFT PANEL – Video + Index
# -------------------------------------------------
with left:
    st.subheader("Step 1: Paste YouTube Link")

    video_url = st.text_input(
        "YouTube video URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )

    video_id = extract_video_id(video_url) if video_url else None

    if video_id:
        st.image(
            f"https://img.youtube.com/vi/{video_id}/0.jpg",
            use_container_width=True
        )

    if st.button("Build Knowledge Index", use_container_width=True):
        if not video_id:
            st.error("Please paste a valid YouTube video link")
            st.stop()

        try:
            with st.spinner("Fetching transcript and building index..."):
                transcript = get_clean_transcript(video_id)
                st.session_state.chain = build_chain(transcript)
                st.session_state.chat_history = []

            st.success("Index ready. Start chatting!")

        except (NoTranscriptFound, TranscriptsDisabled):
            st.session_state.chain = None
            st.session_state.chat_history = []

            st.error(
                "This YouTube video does not have an available transcript, "
                "so I cannot work with it."
            )

        except RequestBlocked:
            st.session_state.chain = None
            st.session_state.chat_history = []

            st.warning(
                "Transcript access is temporarily blocked by YouTube. "
                "Please try again later or use a different video."
            )

        except Exception:
            st.session_state.chain = None
            st.session_state.chat_history = []

            st.error(
                "An unexpected error occurred while processing the video. "
                "Please try again later."
            )

    st.divider()

    if st.session_state.chain:
        if st.button("Reset & Load New Video", use_container_width=True):
            st.session_state.chain = None
            st.session_state.chat_history = []
            st.rerun()


# -------------------------------------------------
# RIGHT PANEL – Chat (INPUT ALWAYS AT BOTTOM)
# -------------------------------------------------
with right:
    st.subheader("Step 2: Chat with the Video")

    if not st.session_state.chain:
        st.info("Build the index first to start chatting.")
    else:
        # Chat input FIRST (Streamlit pins it to bottom)
        question = st.chat_input("Ask something about the video...")

        if question:
            # Save user message
            st.session_state.chat_history.append(
                {"role": "user", "content": question}
            )

            # Generate assistant response
            with st.spinner("Thinking..."):
                answer = st.session_state.chain.invoke(question)

            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )

        # Render full chat history AFTER input
        for chat in st.session_state.chat_history:
            with st.chat_message(chat["role"]):
                st.markdown(chat["content"])
