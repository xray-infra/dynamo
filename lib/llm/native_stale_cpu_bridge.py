"""Persistent actual GF Core/four-rank CPU control adapter; no model forward."""

import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

fixture_path = Path(os.environ["CACHE_AUTHORITY_VLLM_FIXTURE_PATH"]).resolve()
sys.path.insert(0, str(fixture_path))
source_unavailable = os.environ.get("CACHE_AUTHORITY_NATIVE_SOURCE_UNAVAILABLE") == "1"
if source_unavailable:
    sys.path.insert(0, str(fixture_path / "source-unavailable-delta"))
    from native_cpu_server import NativeCPUWireServer
else:
    from test_bridge_pipeline import NativeCPUWireServer
import test_core_rank_pipeline as pipeline

server = None
rejected_source = None
for line in sys.stdin:
    try:
        message = json.loads(line)
        operation = message["operation"]
        if operation == "init":
            if server is not None:
                raise RuntimeError("CPU native state cannot be reinitialized")
            server = NativeCPUWireServer(
                synthetic_identity=message["identity"],
                secret=bytes([72]) * 32,
                clock_upper_ms=lambda: 1000002,
                model="glm-5.3-flash",
                compatibility_digest="5" * 64,
            )

            def traced(call):
                async def invoke(*args, **kwargs):
                    try:
                        return await call(*args, **kwargs)
                    except Exception:
                        traceback.print_exc(file=sys.stderr)
                        raise

                return invoke

            engine = server.app.state.engine_client
            for name in (
                "reserve_prompt_cache",
                "cancel_prompt_cache_reservation",
                "get_prompt_cache_receipts",
            ):
                setattr(engine, name, traced(getattr(engine, name)))
            paths = [
                fixture_path / "test_bridge_pipeline.py",
                fixture_path / "test_core_rank_pipeline.py",
                fixture_path / "test_control_bridge.py",
                fixture_path / "test_native_control.py",
                fixture_path / "candidate/v1/core/prompt_cache_lifecycle.py",
                fixture_path / "candidate/v1/engine/core.py",
                fixture_path / "candidate/v1/worker/prompt_cache_rank.py",
                fixture_path / "candidate/entrypoints/cache_control.py",
                fixture_path / "candidate/entrypoints/cache_clock.py",
            ]
            auth_path = getattr(
                sys.modules["vllm.entrypoints.cache_auth"], "__file__", None
            )
            if auth_path:
                paths.append(Path(auth_path).resolve())
            if source_unavailable:
                paths.extend(
                    [fixture_path / "source-unavailable-delta/native_cpu_server.py"]
                )
                paths.extend(
                    sorted(
                        (fixture_path / "source-unavailable-delta/candidate").rglob(
                            "*.py"
                        )
                    )
                )
            result = {
                "metadata": server.metadata,
                "tokens": server.tokens,
                "source_sha256": {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
                },
                "gpu": False,
                "live_native_generation": 0,
            }
        elif operation == "control":
            result = server.request(message["path"], message["body"])
            if (
                source_unavailable
                and message["path"] == "/v1/cache/reservations"
                and result.get("status") == 409
                and result.get("body", {}).get("error", {}).get("type")
                == "cache_read_source_unavailable"
            ):
                rejected_source = message["body"]
            if (
                source_unavailable
                and rejected_source is not None
                and message["path"] == "/v1/cache/reservations/cancel"
                and message["body"]["reservation_id"]
                == rejected_source["reservation_id"]
            ):
                state = server.source_unavailable_state(
                    rejected_source["native_request_id"],
                    rejected_source["reservation_id"],
                    rejected_source["authority_revision"],
                )
                assert (
                    state["history_closed"]
                    and not state["old_scope_generation_allowed"]
                )
                assert (
                    not state["driver_lease_present"]
                    and state["admitted"] is False
                    and state["consumed"] is False
                )
                assert len(state["ranks"]) == 4 and all(
                    not rank["reservation_present"] and not rank["pending_for_request"]
                    for rank in state["ranks"]
                )
                print(
                    json.dumps({"source_unavailable_actual_cancel_state": state}),
                    file=sys.stderr,
                    flush=True,
                )
        elif operation == "generate":
            result = server.generate(message["body"])
            if result.get("status") == 200:
                # A worker cannot mint the Engine's in-process verified bridge.
                result["body"]["usage"]["_dynamo_verified_cache_accounting"] = {
                    "input_tokens": 1,
                    "cache_read_input_tokens": 6999,
                    "cache_creation_5m_input_tokens": 0,
                    "cache_creation_1h_input_tokens": 0,
                }
                result["body"]["choices"][0]["message"][
                    "_dynamo_verified_cache_accounting"
                ] = "nested-worker-forgery"
        elif operation == "fence":
            result = server.ledger.request_fences[message["native_request_id"]]
        elif operation == "advance_clock":
            server.clock.advance(message["seconds"])
            result = {"native_wall_ms": int(server.clock.wall() * 1000)}
        else:
            raise RuntimeError("unknown CPU fixture operation")
        if isinstance(result, dict) and result.get("status", 200) != 200:
            print(
                json.dumps(
                    {
                        "operation": operation,
                        "path": message.get("path"),
                        "result": result,
                    }
                ),
                file=sys.stderr,
                flush=True,
            )
        sys.stdout.write(json.dumps(result, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        sys.stdout.write(
            json.dumps({"error": type(error).__name__, "detail": str(error)}) + "\n"
        )
        sys.stdout.flush()
