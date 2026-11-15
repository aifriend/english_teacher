import os

# Keep every test offline: no real key, no model downloads, no LLM calls.
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
