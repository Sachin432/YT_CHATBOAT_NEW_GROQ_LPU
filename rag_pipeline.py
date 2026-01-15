"""
rag_pipeline.py

RAG pipeline using Groq LPU-based LLMs for fast, cloud inference.
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
# Prompt Template
# -------------------------------------------------
PROMPT = PromptTemplate(
    template="""
You are a helpful assistant.
Answer ONLY using the provided transcript context.
If the context is insufficient, say "I don't know".

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"],
)


# -------------------------------------------------
# Groq LLM Call
# -------------------------------------------------
def call_groq(prompt: str) -> str:
    """
    Sends a prompt to Groq's LPU-based LLM and returns the response.
    """
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content.strip()


# -------------------------------------------------
# Helper
# -------------------------------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# -------------------------------------------------
# Build RAG Chain
# -------------------------------------------------
def build_chain(transcript_text: str):
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
            }
        )
        | PROMPT
        | RunnableLambda(lambda x: call_groq(x.to_string()))
        | StrOutputParser()
    )

    return chain

