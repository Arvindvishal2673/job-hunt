"""Central configuration loaded from environment / .env file."""

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - dotenv is optional at runtime
    pass

def get_secret(key: str, default: str = "") -> str:
    """Retrieve secret from Streamlit secrets (for cloud deployments) or env vars."""
    try:
        import streamlit as st
        # st.secrets behaves like a dict and can contain the keys
        if hasattr(st, "secrets") and key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return os.getenv(key, default)


GROQ_API_KEY = get_secret("GROQ_API_KEY", "")
GROQ_MODEL = get_secret("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

ADZUNA_APP_ID = get_secret("ADZUNA_APP_ID", "")
ADZUNA_APP_KEY = get_secret("ADZUNA_APP_KEY", "")
APIFY_API_TOKEN = get_secret("APIFY_API_TOKEN", "")


REQUEST_TIMEOUT = 20
MAX_EVALS_DEFAULT = 40
