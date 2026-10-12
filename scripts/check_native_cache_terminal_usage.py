#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the terminal-usage producer gate without the full native HTTP frontend.

Extracts verbatim production functions and regression tests, and uses the
workspace-pinned dynamo-protocols base DTO in place of the Nv wrapper (whose
internal metadata is not carried by the projection). Error construction is also
substituted with anyhow; error classification is not covered. No request dispatch, receipt
verification, SSE codec or GPU/integration coverage is claimed by this gate.
Full-crate converter tests remain registered in native_http_engine.rs.
"""

import argparse
import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path


def function(source: str, name: str, test: bool = False) -> str:
    # These source functions close at column zero (or four spaces in the existing
    # inline test module); nested bodies close more deeply. Fail on absent symbols
    # rather than falling back to a copied implementation.
    match = re.search(
        rf"^(?P<indent> *)(?:pub\(crate\) )?fn {re.escape(name)}\b[\s\S]*?^(?P=indent)}}\s*$",
        source,
        re.MULTILINE,
    )
    if not match:
        raise RuntimeError(f"source function not found: {name}")
    return ("#[test]\n" if test else "") + match.group(0) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    engine_path = root / "lib/llm/src/native_http_engine.rs"
    tests_path = root / "lib/llm/src/native_cache_terminal_usage_tests.rs"
    accounting_path = root / "lib/llm/src/cache_authority/cache_accounting.rs"
    messages_path = root / "lib/llm/src/protocols/anthropic/native_messages.rs"
    paths = [engine_path, tests_path, accounting_path, messages_path]
    sources = {p: p.read_text() for p in paths}
    for path in paths:
        print(
            f"source {path.relative_to(root)} sha256={hashlib.sha256(path.read_bytes()).hexdigest()}",
            flush=True,
        )
    engine = sources[engine_path]
    tests = sources[tests_path]
    pinned_protocol = re.search(
        r'dynamo-protocols\s*=\s*\{\s*version\s*=\s*"([^"]+)"',
        (root / "Cargo.toml").read_text(),
    )
    if pinned_protocol is None:
        raise RuntimeError("workspace dynamo-protocols pin not found")
    version = pinned_protocol.group(1)
    helpers = [
        "strip_unsigned_cache_accounting",
        "attach_verified_cache_accounting",
        "verified_ttl_terminal_usage",
        "native_reasoning_tokens",
    ]
    if "fn native_terminal_usage_template(" in engine:
        helpers[2:2] = ["native_terminal_usage_template", "record_native_usage_chunk"]
    extracted = "\n".join(function(engine, name) for name in helpers)
    extracted_tests = "\n".join(
        function(tests, name) for name in ["native_chunk", "report"]
    )
    for name in re.findall(
        r"^fn (native_cache_terminal_usage_\w+)\(", tests, re.MULTILINE
    ):
        extracted_tests += function(tests, name, test=True)
    # Also execute the pre-existing counter/unknown/reasoning guards unchanged.
    if "fn stream_usage_fixture(" in engine:
        extracted_tests += function(engine, "stream_usage_fixture")
    for name in [
        "native_cache_bridge_risk_terminal_requires_native_counter_agreement",
        "native_cache_usage_keeps_completion_details_with_verified_prefill_accounting",
    ]:
        extracted_tests += function(engine, name, test=True)
    with tempfile.TemporaryDirectory(
        prefix="native-cache-terminal-usage-"
    ) as directory:
        directory = Path(directory)
        (directory / "src").mkdir()
        (directory / "Cargo.toml").write_text(
            f"""[package]
name = "native-cache-terminal-usage-gate"
version = "0.0.0"
edition = "2024"
[workspace]
[dependencies]
anyhow = "1"
serde = {{ version = "1", features = ["derive"] }}
serde_json = "1"
thiserror = "2"
dynamo-protocols = "{version}"
"""
        )
        accounting_include = (
            str(accounting_path).replace("\\", "\\\\").replace('"', '\\"')
        )
        (directory / "src/lib.rs").write_text(
            "".join(
                [
                    '''#![allow(dead_code, unused_imports)]
use serde_json::{Value, json};
use anyhow::Error;
use dynamo_protocols::types::CreateChatCompletionStreamResponse as NvCreateChatCompletionStreamResponse;
const VERIFIED_CACHE_ACCOUNTING_FIELD: &str = "_dynamo_verified_cache_accounting";
fn invalid(message: impl Into<String>) -> Error { anyhow::anyhow!(message.into()) }
mod cache_authority {
    #[derive(Debug, thiserror::Error)]
    pub enum Error { #[error("{0}")] Evidence(&'static str) }
    #[path = "''',
                    accounting_include,
                    """"]
    pub mod cache_accounting;
}
mod native_messages {
    use anyhow::Result;
    use serde_json::{Value, json};
    use crate::cache_authority::cache_accounting::VerifiedCacheAccounting;
    const VERIFIED_CACHE_ACCOUNTING: &str = "_dynamo_verified_cache_accounting";
""",
                    function(sources[messages_path], "usage"),
                    "}\n",
                    extracted,
                    """
#[cfg(test)] mod tests {
use super::*;
use crate::cache_authority::cache_accounting::VerifiedCacheAccounting;
""",
                    extracted_tests,
                    "}\n",
                ]
            )
        )
        command = ["cargo", "test", "--manifest-path", str(directory / "Cargo.toml")]
        if args.offline:
            command.append("--offline")
        return subprocess.run(command, env=os.environ.copy(), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
