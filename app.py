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
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("YouTube Video Chatbot")
st.caption("Ask questions directly from a YouTube video using AI (GROQ_MODEL: llama-3.1-8b-instant)")


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
st.session_state.setdefault("chain", None)
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("video_id", None)
st.session_state.setdefault("index_ready", False)


# -------------------------------------------------
# Layout
# -------------------------------------------------
left, right = st.columns([1.1, 2])


# -------------------------------------------------
# LEFT PANEL – Video & Indexing
# -------------------------------------------------
with left:
    st.subheader("Step 1 · Video Input")

    video_url = st.text_input(
        "Paste YouTube video link",
        placeholder="https://www.youtube.com/watch?v=..."
    )

    video_id = extract_video_id(video_url) if video_url else None

    if video_id:
        st.session_state.video_id = video_id
        st.image(
            f"https://img.youtube.com/vi/{video_id}/0.jpg",
            use_container_width=True
        )

    build_btn = st.button(
        "Build Knowledge Index",
        use_container_width=True,
        disabled=not video_id
    )

    if build_btn:
        try:
            st.session_state.index_ready = False
            st.session_state.chat_history = []

            with st.spinner("Fetching transcript and building semantic index..."):
                transcript = get_clean_transcript(video_id)
                st.session_state.chain = build_chain(transcript)

            st.session_state.index_ready = True
            st.success("Index ready. You can now chat with the video.")

        except (NoTranscriptFound, TranscriptsDisabled):
            st.session_state.chain = None
            st.session_state.index_ready = False

            st.error(
                "This video does not have an available transcript. "
                "Please try a different video."
            )

        except RequestBlocked:
            st.session_state.chain = None
            st.session_state.index_ready = False

            st.warning(
                "YouTube temporarily blocked transcript access for this video. "
                "Please try again later."
            )

        except Exception:
            st.session_state.chain = None
            st.session_state.index_ready = False

            st.error(
                "Something went wrong while processing the video. "
                "Please try again later."
            )

    st.divider()

    if st.session_state.index_ready:
        st.success("Status: Ready to Chat")

    if st.session_state.chain:
        if st.button("Reset & Load New Video", use_container_width=True):
            st.session_state.chain = None
            st.session_state.chat_history = []
            st.session_state.index_ready = False
            st.session_state.video_id = None
            st.rerun()


# -------------------------------------------------
# RIGHT PANEL – Chat Interface
# -------------------------------------------------
with right:
    st.subheader("Step 2 · Chat with the Video")

    if not st.session_state.index_ready:
        st.info(
            "Paste a YouTube link and build the knowledge index "
            "to start asking questions."
        )
    else:
        # Display chat history
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(chat["question"])

            with st.chat_message("assistant"):
                st.markdown(chat["answer"])

        # Chat input
        question = st.chat_input(
            "Ask a question about the video…",
            disabled=not st.session_state.index_ready
        )

        if question:
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing video content..."):
                    answer = st.session_state.chain.invoke(question)
                    st.markdown(answer)

            st.session_state.chat_history.append(
                {"question": question, "answer": answer}
            )

