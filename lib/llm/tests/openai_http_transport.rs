// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[test]
fn upstream_http_transport_preserves_all_protocols_and_request_modes() {
    let script = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../scripts/check_openai_http_frontend.py");
    let output = std::process::Command::new("python3")
        .arg(script)
        .arg("--binary")
        .arg(env!("CARGO_BIN_EXE_openai-http-frontend"))
        .output()
        .expect("execute loopback HTTP fixture");
    assert!(
        output.status.success(),
        "HTTP integration failed:\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["fixture_only"], true);
    assert_eq!(report["passed"], 12);
}
