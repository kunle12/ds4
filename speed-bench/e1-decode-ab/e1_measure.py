#!/usr/bin/env python3
"""E1 A/B decode measurement for the GLM 5.3 pair.

Sends a cold streaming chat completion (prefill + decode), then optionally
repeats the identical request, which the coordinator serves from session cache
so only decode is paid. That gives several decode samples per expensive prefill.

The prompt is a fixed nonce plus a slice of a text file, so the cache can be hit
deliberately on repeats and cannot be hit across runs with different nonces.

Usage:
  e1_measure.py --url URL --model ID --prompt-file F [--bytes N] \
                [--max-tokens 40] [--nonce STR] [--repeats 1] [--out FILE]
Output: one JSON object per request, one per line.
"""
import argparse
import hashlib
import http.client
import json
import os
import sys
import time
import urllib.parse


def build_prompt(path, nbytes):
    with open(path, "rb") as f:
        raw = f.read()
    if nbytes and nbytes > 0:
        if nbytes <= len(raw):
            raw = raw[:nbytes]
        else:
            reps = (nbytes + len(raw) - 1) // len(raw)
            raw = (raw * reps)[:nbytes]
    return raw.decode("utf-8", "ignore")


def request_once(u, path, data, timeout):
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    headers = {"Content-Type": "application/json", "Content-Length": str(len(data))}
    t0 = time.time()
    conn.request("POST", path, body=data, headers=headers)
    resp = conn.getresponse()

    r = {
        "http_status": resp.status,
        "ttft_s": None,
        "decode_wall_s": None,
        "content_chunks": 0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "cached_tokens": None,
        "cache_write_tokens": None,
        "decode_ms_per_token": None,
        "decode_tps": None,
        "gap_ms_min": None,
        "gap_ms_median": None,
        "gap_ms_max": None,
        "text_head": "",
        "error": None,
    }
    gaps = []
    last = None
    first_content = None
    text = []
    usage = None
    try:
        for raw_line in resp:
            line = raw_line.decode("utf-8", "ignore").strip()
            if not line or not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except Exception:
                continue
            if isinstance(obj, dict) and obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []) or []:
                delta = ch.get("delta") or {}
                piece = delta.get("content")
                if not piece:
                    continue
                now = time.time()
                if first_content is None:
                    first_content = now
                    r["ttft_s"] = now - t0
                else:
                    gaps.append((now - last) * 1000.0)
                last = now
                r["content_chunks"] += 1
                text.append(piece)
    except Exception as exc:  # noqa: BLE001
        r["error"] = "%s: %s" % (type(exc).__name__, exc)
    finally:
        conn.close()

    if first_content is not None and last is not None:
        r["decode_wall_s"] = last - first_content
    if usage:
        r["prompt_tokens"] = usage.get("prompt_tokens")
        r["completion_tokens"] = usage.get("completion_tokens")
        details = usage.get("prompt_tokens_details") or {}
        r["cached_tokens"] = details.get("cached_tokens")
        r["cache_write_tokens"] = details.get("cache_write_tokens")
    if gaps:
        sg = sorted(gaps)
        r["gap_ms_min"] = sg[0]
        r["gap_ms_median"] = sg[len(sg) // 2]
        r["gap_ms_max"] = sg[-1]
    if r["decode_wall_s"] and r["completion_tokens"] is not None and r["completion_tokens"] > 1:
        per = r["decode_wall_s"] * 1000.0 / (r["completion_tokens"] - 1)
        r["decode_ms_per_token"] = per
        r["decode_tps"] = 1000.0 / per if per > 0 else None
    elif r["decode_wall_s"] and r["content_chunks"] > 1:
        per = r["decode_wall_s"] * 1000.0 / (r["content_chunks"] - 1)
        r["decode_ms_per_token"] = per
        r["decode_tps"] = 1000.0 / per if per > 0 else None
    r["text_head"] = "".join(text)[:120].replace("\n", "\\n")
    full = "".join(text)
    r["text_len"] = len(full)
    r["text_sha256"] = hashlib.sha256(full.encode("utf-8")).hexdigest()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--bytes", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--nonce", default="")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", default="")
    ap.add_argument("--timeout", type=float, default=3600.0)
    a = ap.parse_args()

    nonce = a.nonce or os.urandom(4).hex()
    body_text = build_prompt(a.prompt_file, a.bytes)
    content = (
        "NONCE-%s\n\n" % nonce
        + body_text
        + "\n\nIgnore the text above. Output exactly the numbers 0 through 39, "
        "one per line, and nothing else.\n"
    )
    payload = {
        "model": a.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": a.max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    data = json.dumps(payload).encode("utf-8")

    u = urllib.parse.urlparse(a.url)
    path = "/v1/chat/completions"

    lines = []
    for i in range(max(1, a.repeats)):
        r = request_once(u, path, data, a.timeout)
        r["nonce"] = nonce
        r["repeat"] = i
        r["prompt_bytes"] = len(body_text)
        out = json.dumps(r)
        print(out, flush=True)
        lines.append(out)
        if r["error"] or r["http_status"] != 200:
            break
    if a.out:
        with open(a.out, "w") as f:
            f.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
