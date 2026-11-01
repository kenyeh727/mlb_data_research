"""Meta Model API client (OpenAI-compatible Chat Completions).

Base URL https://api.meta.ai/v1 and model muse-spark-1.3, corroborated by
multiple third-party docs (OpenAI-compatible, Bearer MODEL_API_KEY).
The key is read ONLY from this subfolder's .env (via dotenv when
available, else a tiny manual parser). It is never hardcoded, logged,
printed, or requested from the user. If no key is present, callers
fall back to template prose and flag that live Muse Spark prose needs
the key.
"""
import json
import os
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_URL = "https://api.meta.ai/v1"
MODEL = "muse-spark-1.3"


def _read_env_file() -> dict:
    values = {}
    path = os.path.join(HERE, ".env")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                values[k.strip()] = v.strip().strip("\"'")
    return values


def get_api_key() -> str | None:
    """Return META_API_KEY from .env or environment, never asking for it."""
    try:
        from dotenv import load_dotenv
        load_dotenv(os.path.join(HERE, ".env"))
    except ImportError:
        pass
    key = os.environ.get("META_API_KEY") or _read_env_file().get("META_API_KEY")
    return key or None


def generate_prose(prompt: str, timeout: int = 60) -> tuple:
    """Call Muse Spark; return (text, error). error is None on success."""
    key = get_api_key()
    if not key:
        return None, "no META_API_KEY in .env"
    payload = json.dumps({
        "model": MODEL,
        "messages": [
            {"role": "system",
             "content": "You are a baseball scouting analyst. Write concise, "
                        "numbers-grounded analysis. Never invent numbers; "
                        "use only the numbers provided."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 600,
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions", data=payload,
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
        return body["choices"][0]["message"]["content"].strip(), None
    except (urllib.error.URLError, KeyError, IndexError, ValueError) as e:
        return None, f"Meta API call failed: {e}"
