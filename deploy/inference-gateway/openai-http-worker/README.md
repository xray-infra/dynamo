<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Existing OpenAI HTTP workers: bounded frontend deployment

`glmflash.json` is the Flash frontend configuration, not a GPU worker launcher.
`compose.yaml` fixes the current serving isolation/resource settings. The public
listener is private-network accessible; tenant headers are not authentication.
Identity/authorization remains the upstream gateway's responsibility.

## Qualification and build context

Build `openai-http-frontend` from committed, locked source in the pinned CPU SDK.
Verify every tracked production source and dependency file, then pass the Cargo
regressions and real Chat/Messages/Responses text/tool unary/stream checks. Also
verify namespace cold/warm/fresh-B behavior (including both salt locations),
header precedence, explicit-key affinity and targeted unsupported-control errors.

The runtime image is `FROM scratch`, not a child of a retired serving image.
Assemble a dedicated context `runtime/` containing:

- `/opt/openai-http-frontend`: the qualified ELF, with recorded SHA256;
- the five ELF `NEEDED` libraries and interpreter at their required paths;
- the public CA certificate bundle and system-library copyright notices.

Resolve source symlinks when copying libraries. Verify installed package checksums
for libc6/libgcc-s1/libstdc++6 on the immutable source image, record their versions
and image ID, and hash every assembled file. Preserve the full package-verification
receipt: any library/content mismatch fails qualification. A minimal source image
may have stripped package NEWS/README documentation; classify only the exact known
missing documentation paths separately, never suppress other findings.
These are system dependencies only;
do not copy launchers, Python packages, cache clients, keys, hooks, weights or
entire library directories. Recheck `readelf` whenever linkage changes. Qualify
**this runtime image** against real workers before switching public traffic.

```sh
docker build --network none \
  --build-arg REVISION="$TESTED_REVISION" \
  --build-arg BINARY_SHA256="$QUALIFIED_BINARY_SHA256" \
  --build-arg RUNTIME_SOURCE_IMAGE="$AUDITED_LIBRARY_SOURCE_IMAGE" \
  -f Dockerfile -t "$CANDIDATE_IMAGE" "$QUALIFIED_CONTEXT"
```

The source library image label records provenance only: no layer from that image
is inherited. Build context manifests and acceptance receipts are deployment
evidence, not executable rollback components.

## Cutover and retirement

Normal PR merge follows passing code and real candidate acceptance; confirm merged
production source bytes equal the tested source before deploying. Set
`FRONTEND_IMAGE` to the qualified image digest and `FRONTEND_REVISION` to the merged
revision. Verify the checked-in JSON equals the intended production configuration.
Compose validation may run on a developer machine when the host has no Compose
plugin. On that host, use the equivalent explicit Docker settings from the
validated definition; installing a daemon/plugin is not a deployment prerequisite.
Do not use `docker compose up` as an implicit unreviewed replacement operation.
Explicitly drain the old owned frontend, start the new definition, and repeat public
acceptance. Only the frontend changes; GPU workers and other models are untouched.

After public acceptance, remove the stopped old frontend and owned canaries.
Delete old binaries/images/configurations only after inspecting all container
references (including directory mounts and descendant image dependencies). Do not
restore retired private/shared-load implementations or keep them as rollback.
Never delete current mounts or running worker sources as a shortcut to migration.

## Scope

Automatic GPU prefix caching, cache namespace mapping, native usage projection,
protocol terminal integrity and explicit OpenAI-key affinity are qualified separately.
Affinity is not real-time KV-residency or load-aware routing. Explicit TTL/retention
controls cannot currently be represented and return targeted unsupported errors.
Cache-creation projection zero does not prove physical writes or write pricing.
Worker provenance/private-hook removal, cancellation resource reclamation, restart/
eviction behavior and optional CPU/BeeGFS tiers require their own qualification.
