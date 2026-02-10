import os

from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-opus-4-6"
MAX_TOKENS = 8192
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
