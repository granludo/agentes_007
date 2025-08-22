#!/usr/bin/env python3

import os
import sys
import json
import argparse
from typing import Optional, Dict, Any

try:
    from openai import OpenAI
except ImportError as import_error:
    sys.stderr.write(
        "Error: The 'openai' package is not installed. Install with: pip install openai\n"
    )
    raise import_error


GRAY = "\033[90m"
RESET = "\033[0m"


def get_env(name: str) -> Optional[str]:
    value = os.getenv(name)
    return value if value and value.strip() else None


def get_key_from_json(path: str) -> Optional[str]:
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


def print_gray_json(label: str, data: Any) -> None:
    try:
        json_text = json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        json_text = str(data)
    sys.stdout.write(f"{GRAY}{label}\n{json_text}{RESET}\n")


def as_serializable(obj: Any) -> Any:
    # Try common serialization paths for OpenAI client objects
    try:
        return obj.model_dump()
    except Exception:
        pass
    try:
        return json.loads(obj.model_dump_json())
    except Exception:
        pass
    try:
        return json.loads(obj.json())
    except Exception:
        pass
    return str(obj)


def build_prompt(system_text: str, transcript: str, user_message: str) -> str:
    if system_text:
        header = f"{system_text}\n\n"
    else:
        header = ""
    return f"{header}{transcript}User: {user_message}\nAssistant:"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Command-line chat using OpenAI Completions API (instruct models)."
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_COMPLETIONS_MODEL", "gpt-3.5-turbo-instruct"),
        help="Completions model (default from OPENAI_COMPLETIONS_MODEL or gpt-3.5-turbo-instruct)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens for each response (default: 256)",
    )
    parser.add_argument(
        "--system",
        default=os.getenv("OPENAI_SYSTEM_INSTRUCTION", "You are a helpful assistant."),
        help="System-style instruction prefixed to the prompt",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print request and response payloads (gray)",
    )

    args = parser.parse_args()

    api_key = get_env("OPENAI_API_KEY") or get_key_from_json("/opt/mykey.json")
    if not api_key:
        sys.stderr.write(
            "Error: OPENAI_API_KEY not found. Set it in env or /opt/mykey.json.\n"
        )
        return 1

    client = OpenAI(api_key=api_key)

    transcript = ""
    sys.stdout.write('Type "exit" to quit.\n')

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            sys.stdout.write("\n")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        prompt = build_prompt(args.system, transcript, user_input)

        request_payload: Dict[str, Any] = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "stop": ["\nUser:", "User:"],
        }

        if args.verbose:
            print_gray_json(
                ">> Request:",
                {
                    "endpoint": "completions.create",
                    **{k: (v if k != "prompt" else (v if len(v) <= 1000 else v[:1000] + "... [truncated]")) for k, v in request_payload.items()},
                },
            )

        try:
            response = client.completions.create(**request_payload)
        except Exception as error:
            sys.stderr.write(f"API error: {error}\n")
            continue

        if args.verbose:
            print_gray_json("<< Response:", as_serializable(response))

        try:
            assistant_text = response.choices[0].text.strip()
        except Exception:
            sys.stderr.write("Unexpected response format.\n")
            continue

        sys.stdout.write(f"Assistant: {assistant_text}\n")

        # Append to transcript for conversational context
        transcript += f"User: {user_input}\nAssistant: {assistant_text}\n"

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
