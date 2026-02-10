import os

from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-opus-4-6"
MAX_TOKENS = 8192
MODEL_API_KEY = os.getenv("MODEL_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
