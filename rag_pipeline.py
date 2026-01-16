"""
rag_pipeline.py

RAG pipeline using Groq LPU-based LLMs
with conversational memory support.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

from groq import Groq

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda
)
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings


# -------------------------------------------------
# Environment Configuration
# -------------------------------------------------
load_dotenv(
    dotenv_path=Path(__file__).resolve().parent / ".env",
    override=True
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")

if not GROQ_API_KEY or not GROQ_MODEL:
    raise RuntimeError("Missing GROQ_API_KEY or GROQ_MODEL in .env")


# -------------------------------------------------
# Groq Client
# -------------------------------------------------
groq_client = Groq(api_key=GROQ_API_KEY)


# -------------------------------------------------
# Embeddings
# -------------------------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={
        "token": os.getenv("HF_TOKEN")
    }
)


# -------------------------------------------------
# Prompt Template (WITH MEMORY)
# -------------------------------------------------
PROMPT = PromptTemplate(
    template="""
You are a helpful assistant.
Use the conversation history and transcript context to answer the question.
Answer ONLY using the transcript context.
If the answer is not present, say "I don't know".

Conversation History:
{history}

Transcript Context:
{context}

Current Question:
{question}
""",
    input_variables=["history", "context", "question"],
)


# -------------------------------------------------
# Groq LLM Call
# -------------------------------------------------
def call_groq(prompt: str) -> str:
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message.content.strip()


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def format_chat_history(chat_history, max_turns: int = 4) -> str:
    """
    Formats recent chat history into a prompt-friendly string.
    Uses last N turns only to keep context bounded.
    """
    if not chat_history:
        return "None"

    recent_history = chat_history[-max_turns:]
    history_text = ""

    for turn in recent_history:
        history_text += f"User: {turn['question']}\n"
        history_text += f"Assistant: {turn['answer']}\n"

    return history_text.strip()


# -------------------------------------------------
# Build RAG Chain (WITH MEMORY)
# -------------------------------------------------
def build_chain(transcript_text: str, chat_history: list):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = splitter.create_documents([transcript_text])

    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    chain = (
        RunnableParallel(
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
                "history": RunnableLambda(
                    lambda _: format_chat_history(chat_history)
                ),
            }
        )
        | PROMPT
        | RunnableLambda(lambda x: call_groq(x.to_string()))
        | StrOutputParser()
    )

    return chain
