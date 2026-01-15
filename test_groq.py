import os
from pathlib import Path
from dotenv import load_dotenv

from groq import Groq

# -------------------------------------------------
# Force-load .env (Windows-safe)
# -------------------------------------------------
load_dotenv(
    dotenv_path=Path(__file__).resolve().parent / ".env",
    override=True
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")

print("GROQ_MODEL =", GROQ_MODEL)

if not GROQ_API_KEY or not GROQ_MODEL:
    raise RuntimeError("Missing GROQ_API_KEY or GROQ_MODEL in .env")

# -------------------------------------------------
# Initialize Groq client
# -------------------------------------------------
client = Groq(api_key=GROQ_API_KEY)

print("\nSending test prompt to Groq...\n")

# -------------------------------------------------
# Send test prompt
# -------------------------------------------------
response = client.chat.completions.create(
    model=GROQ_MODEL,
    messages=[
        {"role": "user", "content": "Say hello in one short sentence."}
    ],
    temperature=0.2
)

# -------------------------------------------------
# Output
# -------------------------------------------------
print("Raw response object:")
print(response)

print("\nModel response:")
print(response.choices[0].message.content.strip())
