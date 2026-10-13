#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# CPU-only native HTTP bridge; no model execution or block-manager GPU service.
# Pin BUILD_IMAGE to an immutable SDK image digest. The SDK's installed toolchain
# may be named "stable", not its numeric version; explicitly select it offline.
: "${BUILD_IMAGE:?set an immutable SDK image digest}"
BUILD_RUST_TOOLCHAIN="${BUILD_RUST_TOOLCHAIN:-stable}"
BUILD_JOBS="${BUILD_JOBS:-8}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
REVISION="$(git rev-parse HEAD)"
mkdir -p "$REPO_ROOT/target"

docker run --rm --runtime=runc --network=none \
    --cpus="$BUILD_JOBS" --memory=20g \
    -e NVIDIA_VISIBLE_DEVICES=void \
    -e RUSTUP_TOOLCHAIN="$BUILD_RUST_TOOLCHAIN" \
    -e CARGO_TARGET_DIR=/target -e CARGO_BUILD_JOBS="$BUILD_JOBS" \
    --label "org.opencontainers.image.revision=$REVISION" \
    -v "$REPO_ROOT:/src:ro" -v "$REPO_ROOT/target:/target" \
    -w /src --entrypoint /bin/bash "$BUILD_IMAGE" -lc \
    'rustc --version; cargo --version; cargo build --release --locked --offline --no-default-features -p dynamo-llm --bin openai-http-frontend'
