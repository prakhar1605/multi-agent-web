#!/usr/bin/env python3
"""Probe the PP API contract empirically, before any adapter is written.

The published docs are ambiguous -- they show ``content`` as a plain string,
list a different Qwen identifier than the one we were given, and say nothing
about images. Rather than guess, this asks the API directly:

    a. which endpoint path works, and whether /v1 is required
    b. which auth header the server accepts
    c. the exact model identifier string
    d. whether the model accepts image input, and in what format

The image question is the one that matters: a browser agent that cannot see a
screenshot is not a browser agent. Better to learn that here than after an
adapter exists.

    python scripts/smoke_api.py
    python scripts/smoke_api.py --model qwen3.5-27b

Credentials come from PPAPI_KEY and PPAPI_BASE_URL in the environment (a .env
at the repo root is loaded automatically). The key is never printed.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import httpx  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from multi_agent_web.config import load_env_file  # noqa: E402

TIMEOUT = 120.0
RULE = "=" * 74


def redact(text: str, key: str) -> str:
    """Belt and braces: strip the key if it ever appears in echoed output."""
    if key and key in text:
        text = text.replace(key, "<PPAPI_KEY redacted>")
    return text


def show(label: str, response: httpx.Response, key: str, limit: int = 1400) -> None:
    print(f"  {label}: HTTP {response.status_code}")
    body = redact(response.text, key)
    if len(body) > limit:
        body = body[:limit] + f"... [{len(response.text) - limit} more chars]"
    print(f"  body: {body}")


def test_image() -> str:
    """A small, unambiguous image: a red circle and the text 'K7'.

    Specific enough that a describing model cannot bluff it -- if the reply
    says red, circle and K7, the pixels genuinely reached the model.
    """
    img = Image.new("RGB", (320, 200), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse([40, 40, 160, 160], fill="red")
    draw.text((200, 90), "K7", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main() -> int:
    load_env_file()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=os.environ.get("PPAPI_MODEL", "qwen3.5-27b"))
    args = parser.parse_args()

    key = os.environ.get("PPAPI_KEY", "").strip()
    base = os.environ.get("PPAPI_BASE_URL", "").strip()
    if not key or not base:
        print("PPAPI_KEY and PPAPI_BASE_URL must both be set (.env or environment).")
        return 2

    if not base.startswith(("http://", "https://")):
        base = "https://" + base
    base = base.rstrip("/")
    # The base URL may or may not already carry /v1; normalise to a bare root.
    root = base[: -len("/v1")] if base.endswith("/v1") else base

    print(RULE)
    print(f"base URL : {base}")
    print(f"root     : {root}")
    print(f"model    : {args.model}")
    print(f"key      : present, {len(key)} chars (not shown)")
    print(RULE)

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=TIMEOUT, follow_redirects=True) as client:
        # --- (c) which model identifiers exist ----------------------------
        print("\n[1] GET /v1/models  -- resolve the exact model identifier")
        model_ids: list[str] = []
        try:
            resp = client.get(f"{root}/v1/models", headers=headers)
            print(f"  HTTP {resp.status_code}")
            if resp.status_code == 200:
                payload = resp.json()
                items = payload.get("data", payload if isinstance(payload, list) else [])
                model_ids = [
                    m.get("id", "") for m in items if isinstance(m, dict)
                ] or [str(m) for m in items]
                print(f"  {len(model_ids)} models available")
                matches = [m for m in model_ids if "qwen" in m.lower()]
                print(f"  qwen matches: {matches or '(none)'}")
                if args.model in model_ids:
                    print(f"  >>> {args.model!r} IS listed")
                else:
                    print(f"  >>> {args.model!r} is NOT in the list")
            else:
                show("models", resp, key)
        except Exception as exc:
            print(f"  request failed: {type(exc).__name__}: {exc}")

        # --- (a) endpoint path --------------------------------------------
        print("\n[2] Text request -- probe /v1/chat/completions vs /chat/completions")
        text_body = {
            "model": args.model,
            "messages": [{"role": "user", "content": "Reply with exactly: PONG"}],
            "max_tokens": 16,
            "temperature": 0,
        }
        working_url: str | None = None
        for path in ("/v1/chat/completions", "/chat/completions"):
            url = f"{root}{path}"
            print(f"\n  -> POST {url}")
            try:
                resp = client.post(url, headers=headers, json=text_body)
            except Exception as exc:
                print(f"  request failed: {type(exc).__name__}: {exc}")
                continue
            show("result", resp, key)
            if resp.status_code == 200 and working_url is None:
                working_url = url
                try:
                    content = resp.json()["choices"][0]["message"]["content"]
                    print(f"  >>> content: {content!r}")
                    print(f"  >>> usage:   {resp.json().get('usage')}")
                except Exception:
                    pass

        if working_url is None:
            print("\nNo working chat endpoint. Stopping before the image test.")
            return 1
        print(f"\n  >>> working endpoint: {working_url}")

        # --- (d) THE critical question: images ----------------------------
        print("\n[3] Image request -- does this model accept a screenshot?")
        b64 = test_image()
        print(f"  test image: 320x200 PNG, red circle + text 'K7', {len(b64)} b64 chars")

        variants = {
            "openai data-url": {
                "model": args.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image in one short sentence: "
                                "what shape, what colour, and what text does it show?",
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64}"},
                            },
                        ],
                    }
                ],
                "max_tokens": 100,
                "temperature": 0,
            },
            "bare base64 image_url": {
                "model": args.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What shape and colour is this?"},
                            {"type": "image_url", "image_url": {"url": b64}},
                        ],
                    }
                ],
                "max_tokens": 100,
                "temperature": 0,
            },
        }

        vision_ok = False
        for name, body in variants.items():
            print(f"\n  -> variant: {name}")
            try:
                resp = client.post(working_url, headers=headers, json=body)
            except Exception as exc:
                print(f"  request failed: {type(exc).__name__}: {exc}")
                continue
            show("result", resp, key)
            if resp.status_code == 200:
                try:
                    content = resp.json()["choices"][0]["message"]["content"]
                except Exception:
                    continue
                print(f"  >>> content: {content!r}")
                lowered = (content or "").lower()
                saw = [w for w in ("red", "circle", "k7") if w in lowered]
                print(f"  >>> matched cues {saw} of ['red', 'circle', 'k7']")
                if len(saw) >= 2:
                    vision_ok = True
                    print(f"  >>> VISION WORKS via {name!r}")
                else:
                    print("  >>> 200 OK but the description does not match the image "
                          "-- the model likely never saw it")

        print("\n" + RULE)
        print(f"endpoint      : {working_url}")
        print(f"auth          : Authorization: Bearer <key>")
        print(f"model         : {args.model}"
              f"{' (listed)' if args.model in model_ids else ' (NOT listed by /v1/models)'}")
        print(f"image support : {'YES' if vision_ok else 'NO / UNCONFIRMED'}")
        print(RULE)
        if not vision_ok:
            print(
                "\nA browser agent needs to see screenshots. If images are genuinely\n"
                "unsupported on this model, stop and pick a vision-capable one before\n"
                "any adapter is written."
            )
        return 0 if vision_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
