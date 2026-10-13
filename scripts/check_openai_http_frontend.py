#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded HTTP integration test of the real upstream frontend, not a live GPU test.

A loopback worker fixture supplies declared native counters and fragmented SSE.
No signed receipts, private cache authority, discovery service or GPU is used.
Run after cargo build --no-default-features -p dynamo-llm --bin openai-http-frontend.
"""

import argparse
import json
import socket
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODEL = "glm-5.3-flash"
USAGE = {
    "prompt_tokens": 100,
    "completion_tokens": 7,
    "total_tokens": 107,
    "prompt_tokens_details": {"cached_tokens": 60, "created_cache_tokens": 24},
    "completion_tokens_details": {"reasoning_tokens": 2},
}
EXPECTED_USAGE = {
    **USAGE,
    "prompt_tokens_details": {"cached_tokens": 60, "cache_write_tokens": 24},
}


class Worker(BaseHTTPRequestHandler):
    requests = []

    def log_message(self, *_args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert self.path == "/v1/chat/completions"
        assert body["model"] == MODEL
        assert body["stream"] is True
        assert body["stream_options"]["include_usage"] is True
        # Public header overrides the body salt; the worker owns cache_salt.
        expected_salt = (
            "root-tenant"
            if body.get("prompt_cache_key") == "responses-top-level-fixture"
            else "tenant-fixture"
        )
        assert body["cache_salt"] == expected_salt, body
        Worker.requests.append((body.get("prompt_cache_key"), self.server.server_port))
        tool = bool(body.get("tools"))
        identity = {
            "id": "chatcmpl-fixture",
            "object": "chat.completion.chunk",
            "created": 7,
            "model": MODEL,
        }
        delta = {"role": "assistant", "content": None if tool else "OK"}
        if tool:
            delta["tool_calls"] = [
                {
                    "index": 0,
                    "id": "call_native_1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"text":"OK"}'},
                }
            ]
        chunks = [
            dict(
                identity, choices=[{"index": 0, "delta": delta, "finish_reason": None}]
            ),
            dict(
                identity,
                choices=[
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls" if tool else "stop",
                    }
                ],
            ),
            dict(identity, choices=[], usage=USAGE),
        ]
        # Mixed LF/CRLF plus arbitrarily split UTF-8/SSE transport boundaries.
        payload = (
            "".join(
                "data: " + json.dumps(chunk) + ("\r\n\r\n" if i == 0 else "\n\n")
                for i, chunk in enumerate(chunks)
            )
            + "data: [DONE]\n\n"
        )
        raw = payload.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        for start in range(0, len(raw), 17):
            self.wfile.write(raw[start : start + 17])
            self.wfile.flush()


def request(url, body=None, headers=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    with urllib.request.urlopen(req, timeout=20) as response:
        return response.read().decode()


def events(raw):
    return [
        line.removeprefix("data: ")
        for line in raw.splitlines()
        if line.startswith("data: ")
    ]


def check_chat(raw, streaming, tool):
    if streaming:
        data = events(raw)
        assert data.count("[DONE]") == 1, raw
        chunks = [json.loads(value) for value in data if value != "[DONE]"]
        usage = [chunk["usage"] for chunk in chunks if chunk.get("usage")]
        assert len(usage) == 1, raw
        choices = [choice for chunk in chunks for choice in chunk["choices"]]
        assert (
            sum(choice.get("finish_reason") is not None for choice in choices) == 1
        ), raw
        text = "".join(choice["delta"].get("content") or "" for choice in choices)
        calls = [
            call
            for choice in choices
            for call in choice["delta"].get("tool_calls") or []
        ]
    else:
        response = json.loads(raw)
        usage = [response["usage"]]
        message = response["choices"][0]["message"]
        text = message.get("content") or ""
        calls = message.get("tool_calls") or []
    assert usage[0] == EXPECTED_USAGE, usage
    if tool:
        assert calls[0]["id"] == "call_native_1", calls
        assert calls[0]["function"]["name"] == "echo", calls
        assert json.loads(
            "".join(call["function"].get("arguments", "") for call in calls)
        ) == {"text": "OK"}
    else:
        assert text == "OK", text


def check_messages(raw, streaming, tool):
    if streaming:
        wire = events(raw)
        # Upstream's shared disconnect monitor may add an OpenAI trailer.
        # It is not a substitute for the native message_delta/message_stop.
        assert wire.count("[DONE]") <= 1, raw
        if "[DONE]" in wire:
            assert wire[-1] == "[DONE]", raw
        data = [json.loads(value) for value in wire if value != "[DONE]"]
        kinds = [event["type"] for event in data]
        assert "error" not in kinds, raw
        assert kinds.count("message_stop") == 1, raw
        assert kinds.count("message_delta") == 1, raw
        usage = next(
            event["usage"] for event in data if event["type"] == "message_delta"
        )
        blocks = [
            event["content_block"]
            for event in data
            if event["type"] == "content_block_start"
        ]
        deltas = [
            event["delta"] for event in data if event["type"] == "content_block_delta"
        ]
        text = "".join(delta.get("text", "") for delta in deltas)
        arguments = "".join(delta.get("partial_json", "") for delta in deltas)
    else:
        response = json.loads(raw)
        usage, blocks = response["usage"], response["content"]
        text = "".join(block.get("text", "") for block in blocks)
        arguments = None
    assert usage["input_tokens"] == 16, usage
    assert usage["output_tokens"] == 7, usage
    assert usage["cache_read_input_tokens"] == 60, usage
    assert usage["cache_creation_input_tokens"] == 24, usage
    if tool:
        block = next(block for block in blocks if block["type"] == "tool_use")
        assert block["id"] and block["name"] == "echo", block
        assert (json.loads(arguments) if streaming else block["input"]) == {
            "text": "OK"
        }
    else:
        assert text == "OK", text


def check_responses(raw, streaming, tool):
    if streaming:
        wire = events(raw)
        data = [json.loads(value) for value in wire if value != "[DONE]"]
        assert not any(event["type"] in ["error", "response.failed"] for event in data)
        completed = [event for event in data if event["type"] == "response.completed"]
        assert len(completed) == 1, raw
        response = completed[0]["response"]
    else:
        response = json.loads(raw)
    assert response["status"] == "completed", response
    usage = response["usage"]
    assert usage["input_tokens"] == 100, usage
    assert usage["output_tokens"] == 7, usage
    assert usage["total_tokens"] == 107, usage
    assert usage["input_tokens_details"]["cached_tokens"] == 60, usage
    assert usage["input_tokens_details"]["cache_write_tokens"] == 24, usage
    if tool:
        call = next(
            item for item in response["output"] if item["type"] == "function_call"
        )
        assert call["call_id"] and call["name"] == "echo", call
        assert json.loads(call["arguments"]) == {"text": "OK"}, call
    else:
        text = "".join(
            part["text"]
            for item in response["output"]
            if item["type"] == "message"
            for part in item["content"]
            if part["type"] == "output_text"
        )
        assert text == "OK", text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    workers = [ThreadingHTTPServer(("127.0.0.1", 0), Worker) for _ in range(2)]
    for worker in workers:
        threading.Thread(target=worker.serve_forever, daemon=True).start()
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
    results = []
    with tempfile.TemporaryDirectory(
        prefix="pr15-wire-", dir=args.binary.resolve().parent
    ) as temporary:
        config = Path(temporary) / "config.json"
        config.write_text(
            json.dumps(
                {
                    "listen": f"127.0.0.1:{port}",
                    "model": MODEL,
                    "workers": [
                        f"http://127.0.0.1:{worker.server_port}" for worker in workers
                    ],
                }
            )
        )
        with (Path(temporary) / "frontend.log").open("w+") as log:
            process = subprocess.Popen(
                [str(args.binary.resolve()), str(config)], stdout=log, stderr=log
            )
            base = f"http://127.0.0.1:{port}"
            try:
                for _ in range(100):
                    assert process.poll() is None, "frontend exited during startup"
                    try:
                        request(base + "/health")
                        break
                    except (urllib.error.URLError, TimeoutError):
                        time.sleep(0.1)
                else:
                    raise RuntimeError("frontend never became healthy")
                schema = {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                }
                for protocol in ["chat", "messages", "responses"]:
                    for tool in [False, True]:
                        for streaming in [False, True]:
                            body = {
                                "model": MODEL,
                                "messages": [{"role": "user", "content": "Return OK"}],
                                "max_tokens": 16,
                                "stream": streaming,
                                "nvext": {"cache_salt": "body-tenant"},
                            }
                            if protocol == "chat":
                                if streaming:
                                    body["stream_options"] = {"include_usage": True}
                                if tool:
                                    body.update(
                                        tools=[
                                            {
                                                "type": "function",
                                                "function": {
                                                    "name": "echo",
                                                    "parameters": schema,
                                                },
                                            }
                                        ],
                                        tool_choice={
                                            "type": "function",
                                            "function": {"name": "echo"},
                                        },
                                    )
                                check = check_chat
                                path = "/v1/chat/completions"
                            elif protocol == "responses":
                                body["input"] = "Return OK"
                                body["max_output_tokens"] = body.pop("max_tokens")
                                body.pop("messages")
                                if tool:
                                    body.update(
                                        tools=[
                                            {
                                                "type": "function",
                                                "name": "echo",
                                                "parameters": schema,
                                            }
                                        ],
                                        tool_choice={
                                            "type": "function",
                                            "name": "echo",
                                        },
                                    )
                                check = check_responses
                                path = "/v1/responses"
                            else:
                                if tool:
                                    body.update(
                                        tools=[
                                            {"name": "echo", "input_schema": schema}
                                        ],
                                        tool_choice={"type": "tool", "name": "echo"},
                                    )
                                check = check_messages
                                path = "/v1/messages"
                            raw = request(
                                base + path,
                                body,
                                headers={"x-tenant-id": "tenant-fixture"},
                            )
                            check(raw, streaming, tool)
                            results.append(
                                {
                                    "protocol": protocol,
                                    "tool": tool,
                                    "stream": streaming,
                                    "passed": True,
                                }
                            )
                body = {
                    "model": MODEL,
                    "messages": [{"role": "user", "content": "Return OK"}],
                    "max_tokens": 16,
                    "nvext": {"cache_salt": "body-tenant"},
                    "prompt_cache_key": "stable-prefix",
                }
                headers = {"x-tenant-id": "tenant-fixture"}
                for _ in range(8):
                    check_chat(
                        request(base + "/v1/chat/completions", body, headers),
                        False,
                        False,
                    )
                destinations = {
                    port for key, port in Worker.requests if key == "stable-prefix"
                }
                assert len(destinations) == 1, destinations

                responses = {
                    "model": MODEL,
                    "input": "Return OK",
                    "max_output_tokens": 16,
                    "nvext": {"cache_salt": "body-tenant"},
                    "prompt_cache_key": "responses-stable-prefix",
                }
                for _ in range(8):
                    response = json.loads(
                        request(base + "/v1/responses", responses, headers)
                    )
                    assert response["status"] == "completed", response
                    assert response["output"][0]["content"][0]["text"] == "OK", response
                destinations = {
                    port
                    for key, port in Worker.requests
                    if key == "responses-stable-prefix"
                }
                assert len(destinations) == 1, destinations

                legacy_namespace = dict(
                    responses,
                    cache_salt="root-tenant",
                    prompt_cache_key="responses-top-level-fixture",
                )
                legacy_namespace.pop("nvext")
                response = json.loads(request(base + "/v1/responses", legacy_namespace))
                assert response["status"] == "completed", response

                def rejected(path, candidate, status):
                    before = len(Worker.requests)
                    try:
                        request(base + path, candidate, headers)
                    except urllib.error.HTTPError as error:
                        payload = json.loads(error.read())
                        assert error.code == status, (error.code, payload)
                        assert isinstance(payload["error"]["message"], str), payload
                        assert payload.get("usage") is None, payload
                    else:
                        raise AssertionError(f"Unsupported request accepted: {path}")
                    assert (
                        len(Worker.requests) == before
                    ), "Rejected request reached worker"

                rejected("/v1/chat/completions", dict(body, model="missing-model"), 404)
                rejected(
                    "/v1/chat/completions",
                    dict(body, prompt_cache_retention="24h"),
                    400,
                )
                rejected(
                    "/v1/responses", dict(responses, prompt_cache_retention="24h"), 400
                )
                rejected(
                    "/v1/messages",
                    dict(body, cache_control={"type": "ephemeral", "ttl": "1h"}),
                    400,
                )
                blocks = dict(
                    body,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "OK",
                                    "cache_control": {"type": "ephemeral"},
                                }
                            ],
                        }
                    ],
                )
                rejected("/v1/messages", blocks, 400)
                print(
                    json.dumps(
                        {
                            "fixture_only": True,
                            "passed": len(results),
                            "cases": results,
                        },
                        indent=2,
                    )
                )
            except BaseException:
                log.flush()
                log.seek(0)
                print(log.read()[-4000:])
                raise
            finally:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                for worker in workers:
                    worker.shutdown()


if __name__ == "__main__":
    main()
