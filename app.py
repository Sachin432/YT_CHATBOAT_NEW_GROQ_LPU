import streamlit as st
from urllib.parse import urlparse, parse_qs

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
st.caption("Ask questions directly from a YouTube video using local AI")

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

        with st.spinner("Fetching transcript and building index..."):
            transcript = get_clean_transcript(video_id)
            st.session_state.chain = build_chain(transcript)
            st.session_state.chat_history = []

        st.success("Index ready. Start chatting!")

    st.divider()

    if st.session_state.chain:
        if st.button("Reset & Load New Video", use_container_width=True):
            st.session_state.chain = None
            st.session_state.chat_history = []
            st.rerun()

# -------------------------------------------------
# RIGHT PANEL – Chat (FULL PERSISTENT CHAT)
# -------------------------------------------------
with right:
    st.subheader("Step 2: Chat with the Video")

    if not st.session_state.chain:
        st.info("Build the index first to start chatting.")
    else:
        # Render full chat history
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(chat["question"])

            with st.chat_message("assistant"):
                st.markdown(chat["answer"])

        # Chat input (fixed at bottom)
        question = st.chat_input("Ask something about the video...")

        if question:
            # Show user message instantly
            with st.chat_message("user"):
                st.markdown(question)

            # Generate and show assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = st.session_state.chain.invoke(question)
                    st.markdown(answer)

            # Save conversation
            st.session_state.chat_history.append(
                {"question": question, "answer": answer}
            )

