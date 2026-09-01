#!/usr/bin/env python3
"""Launcher-stub test for the test-only GLM53_TEST_DROP_ALLOC_CONF hook
(scratch branch test/alloc-iso; PROTOCOL-store-overhead-isolation.md §2).

Directions covered:
  H1 default boot keeps stock PYTORCH_CUDA_ALLOC_CONF on BOTH ranks.
  H2 GLM53_TEST_DROP_ALLOC_CONF=1 with knob=0 drops the env on BOTH ranks,
     announces the drop in the boot log, and leaves serve argv untouched
     (arm A vs arm B argv parity is an env-only difference).
  H3 GLM53_TEST_DROP_ALLOC_CONF=1 with GLM53_KV_OFFLOAD=1 dies fail-closed
     (diagnostics-only flag is legal only on knob=0 boots).

Drives the shipped start.sh via the Harness from test_launcher_kv_offload
(docker/ssh/scp stubbed; nothing talks to a real host).

Run:  python3 tests/test_alloc_iso_hook.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_launcher_kv_offload import (  # noqa: E402
    FAILURES,
    Harness,
    _env_of_head,
    _env_of_worker,
    _launch,
    check,
)

ALLOC = "PYTORCH_CUDA_ALLOC_CONF"
STOCK = "expandable_segments:True"
DROP_LOG = "dropped on both ranks"


def _serve_argv(head_run: list[str]) -> list[str]:
    # strip every `-e KEY=val` pair; what remains (docker flags + image + serve
    # command) must be identical between arm A and arm B boots
    out, skip = [], False
    for tok in head_run:
        if skip:
            skip = False
            continue
        if tok == "-e":
            skip = True
            continue
        out.append(tok)
    return out


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        h = Harness(Path(td))

        print("H1: default keeps stock allocator env on both ranks")
        _, head_run, worker_cmd = _launch(h)
        he, we = _env_of_head(head_run), _env_of_worker(worker_cmd)
        check(he.get(ALLOC) == STOCK, f"H1a head {ALLOC}={he.get(ALLOC)!r}")
        check(we.get(ALLOC) == STOCK, f"H1b worker {ALLOC}={we.get(ALLOC)!r}")
        argv_default = head_run

        print("H2: TEST_DROP_ALLOC_CONF=1 + knob=0 drops env, logs, same argv")
        r = h.run(
            ["eval", "MODEL_DIR=/models/glm DFLASH_MODEL_DIR=/models/dflash; write_inner_scripts; launch_cluster"],
            entry="start.fn.sh",
            GLM53_KV_OFFLOAD="0",
            GLM53_TEST_DROP_ALLOC_CONF="1",
        )
        check(r.returncode == 0, f"H2a launch_cluster proceeds (rc={r.returncode})")
        calls = h.calls()
        head_run2 = next(
            c for c in calls if c[0] == "docker" and c[1] == "run" and "glm53-exl3-head" in c
        )
        worker_cmd2 = next(c[-1] for c in calls if c[0] == "ssh" and "docker run" in c[-1])
        he2, we2 = _env_of_head(head_run2), _env_of_worker(worker_cmd2)
        check(ALLOC not in he2, f"H2b head env dropped (got {he2.get(ALLOC)!r})")
        check(ALLOC not in we2, f"H2c worker env dropped (got {we2.get(ALLOC)!r})")
        check(DROP_LOG in (r.stdout + r.stderr), "H2d drop announced in boot log")
        check(
            _serve_argv(head_run2)[-len(_serve_argv(argv_default)):]
            == _serve_argv(argv_default)[-len(_serve_argv(argv_default)):],
            "H2e serve argv identical to the default boot (env-only difference)",
        )
        # implicit knob=0 (flag alone) behaves the same
        r = h.run(
            ["eval", "MODEL_DIR=/models/glm DFLASH_MODEL_DIR=/models/dflash; write_inner_scripts; launch_cluster"],
            entry="start.fn.sh",
            GLM53_TEST_DROP_ALLOC_CONF="1",
        )
        check(r.returncode == 0 and DROP_LOG in (r.stdout + r.stderr), "H2f implicit knob=0 also drops")

        print("H3: TEST_DROP_ALLOC_CONF=1 + knob=1 refused fail-closed")
        r = h.run(
            ["eval", "MODEL_DIR=/models/glm DFLASH_MODEL_DIR=/models/dflash; write_inner_scripts; launch_cluster"],
            entry="start.fn.sh",
            GLM53_KV_OFFLOAD="1",
            GLM53_TEST_DROP_ALLOC_CONF="1",
        )
        check(r.returncode != 0, f"H3a nonzero exit (rc={r.returncode})")
        check(
            "diagnostics-only flag for knob=0 boots" in (r.stderr + r.stdout),
            "H3b die names the rule",
        )

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
