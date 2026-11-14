import os 
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

EMBEDDING_MODEL = "openai/text-embeddings-3-small"

LLM_MODEL = "openai/gpt-4o-mini"