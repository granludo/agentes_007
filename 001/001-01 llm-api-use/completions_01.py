#!/usr/bin/env python3

import os
import sys
import json
from typing import Optional

try:
    from openai import OpenAI
except ImportError as import_error:
    sys.stderr.write(
        "Error: The 'openai' package is not installed. Install with: pip install openai\n"
    )
    raise import_error


def get_env(name: str) -> Optional[str]:
    """Return environment variable value if set and non-empty, else None."""
    value = os.getenv(name)
    return value if value and value.strip() else None


def get_key_from_json(path: str) -> Optional[str]:
    """Return OPENAI_API_KEY from a JSON file if available, else None."""
    try:
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        key = data.get("OPENAI_API_KEY")
        if key and isinstance(key, str) and key.strip():
            return key.strip()
        return None
    except Exception:
        return None


def main() -> int:
    if len(sys.argv) < 2:
        sys.stderr.write(
            "Usage: completions_01.py \"<message>\"\n"
            "- The message is read from $1 (first CLI argument).\n"
        )
        return 1

    message = sys.argv[1]

    api_key = get_env("OPENAI_API_KEY") or get_key_from_json("/opt/mykey.json")
    if not api_key:
        sys.stderr.write(
            "Error: OPENAI_API_KEY not found. Set it in env or /opt/mykey.json.\n"
        )
        return 1

    # Initialize client (reads API key from env by default, explicit for clarity)
    client = OpenAI(api_key=api_key)

    # Use an instruct model for the legacy Completions API
    model = os.getenv("OPENAI_COMPLETIONS_MODEL", "gpt-3.5-turbo-instruct")

    try:
        response = client.completions.create(
            model=model,
            prompt=message,
            max_tokens=256,
            temperature=0.7,
        )
    except Exception as error:  # noqa: BLE001 - surface any client/network/API error
        sys.stderr.write(f"API error: {error}\n")
        return 2

    try:
        text = response.choices[0].text.strip()
    except Exception:
        sys.stderr.write("Unexpected response format.\n")
        return 2

    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
