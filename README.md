# YouTube Video Chatbot using RAG (Groq LPU)

🔗 **Live Application:**  
https://ytchatboatnewgroqlpu-aphmzmg6jmdb5bmkeozxrf.streamlit.app/

---

## Business Problem

YouTube videos contain a vast amount of valuable knowledge, especially in education, technical training, and research.  
However, extracting specific information from long videos is time-consuming and inefficient, as users must manually scan or rewatch large portions of content.

This project addresses the problem by enabling **direct, question-driven access to YouTube video content**, allowing users to ask natural language questions and receive precise, context-aware answers derived strictly from the video transcript.

---

## Solution Overview

This project implements a **Retrieval-Augmented Generation (RAG) based YouTube chatbot** that combines semantic search with large language models to provide accurate, grounded answers from video transcripts.

Instead of relying on the model’s general knowledge, the system:
- Retrieves only the most relevant transcript segments
- Passes them as context to the LLM
- Generates responses strictly grounded in the video content

---

## System Architecture

High-level pipeline:

1. User provides a YouTube video link
2. Transcript is fetched using YouTube Transcript API
3. Transcript is split into semantic chunks
4. Sentence Transformer generates vector embeddings
5. FAISS performs similarity-based retrieval
6. Relevant context is injected into a prompt
7. Groq LPU-based LLM generates the final answer
8. LangSmith tracks the full execution for observability

---

## Key Features

- Contextual Q&A over YouTube videos  
- Retrieval-Augmented Generation (RAG) architecture  
- Semantic search using vector embeddings  
- Ultra-low latency LLM inference using Groq LPU  
- Interactive Streamlit-based chat interface  
- Persistent chat history per video  
- Built-in observability using LangSmith  

---

## Technologies Used

- **Programming Language:** Python  
- **Frameworks & Libraries:**  
  - LangChain  
  - Streamlit  
  - FAISS  
  - Sentence-Transformers  
- **LLM Provider:** Groq LPU (LLaMA 3.1)  
- **Embeddings Model:** all-MiniLM-L6-v2  
- **Transcript Source:** YouTube Transcript API  
- **Observability:** LangSmith  

---

## Why Sentence Transformers Are Used

Sentence Transformers convert transcript chunks and user questions into dense semantic vectors.  
This enables the system to retrieve relevant content based on meaning rather than keyword matching, which is critical for accurate retrieval in RAG systems.

---

## Why Groq LPU

Groq LPU is used exclusively for the **generation layer** due to:
- Extremely low inference latency
- High throughput for real-time chat applications
- Cloud-based execution suitable for deployment

Retrieval and embedding remain local and deterministic, ensuring modular and scalable design.

---

## Observability with LangSmith

LangSmith provides full visibility into:
- Retrieved transcript chunks
- Prompt construction
- LLM calls and latency
- End-to-end query traces

This makes the RAG pipeline debuggable, transparent, and production-ready.

---

## Project Structure

```text
├── app.py                  # Streamlit UI
├── rag_pipeline.py         # RAG pipeline (retrieval + generation)
├── transcript_utils.py     # Transcript fetching and cleaning
├── test_groq.py            # Groq LLM connectivity test
├── requirements.txt
└── README.md

