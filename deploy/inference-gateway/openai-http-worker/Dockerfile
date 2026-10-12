# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The qualified context contains only the frontend ELF and its audited system
# runtime dependencies. No serving image, GPU runtime or retired private layer
# is inherited. Context assembly and checksum gates are documented in README.md.
FROM scratch
ARG REVISION
ARG BINARY_SHA256
ARG RUNTIME_SOURCE_IMAGE
LABEL org.opencontainers.image.source="https://github.com/xray-infra/dynamo" \
      org.opencontainers.image.revision="${REVISION}" \
      io.xray.frontend.binary.sha256="${BINARY_SHA256}" \
      io.xray.runtime.libraries.source-image="${RUNTIME_SOURCE_IMAGE}"
COPY runtime/ /
USER 1001:1001
ENTRYPOINT ["/opt/openai-http-frontend"]
CMD ["/opt/public.json"]
