import os
from typing import Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Look up a secret: Streamlit secrets first, then environment / .env."""
    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            try:
                if key in st.secrets:
                    return str(st.secrets[key])
            except Exception:
                pass
    except ImportError:
        pass
    return os.environ.get(key, default)
