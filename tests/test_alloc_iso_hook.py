#!/usr/bin/env python3
"""Launcher-stub test for the test-only GLM53_TEST_DROP_ALLOC_CONF hook
(scratch branch test/alloc-iso; PROTOCOL-store-overhead-isolation.md §2).

Directions covered (rev 2, post-Codex CODEX-ALLOC-ISO-HOOK.md — findings 1-5
adopted):
  H1 default boot keeps stock PYTORCH_CUDA_ALLOC_CONF on BOTH ranks.
  H2 GLM53_TEST_DROP_ALLOC_CONF=1 with knob=0 drops the env on BOTH ranks,
     announces the drop in the boot log, and changes NOTHING else: inner rank
     scripts byte-identical to the default boot, head docker-run argv exactly
     equal after removing only the allocator -e pair, and each rank's full env
     map equal to the default boot's minus only that key.
  H3 fail-closed: =1 with GLM53_KV_OFFLOAD=1 is refused PRE-STOP on restart
     (rc=2, zero docker/ssh calls); junk values (2/yes/empty) likewise.

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
    _inner_scripts,
    _launch,
    check,
)

ALLOC = "PYTORCH_CUDA_ALLOC_CONF"
STOCK = "expandable_segments:True"
DROP_LOG = "dropped on both ranks"
LAUNCH = "MODEL_DIR=/models/glm DFLASH_MODEL_DIR=/models/dflash; write_inner_scripts; launch_cluster"


def _strip_alloc_pair(head_run: list[str]) -> list[str]:
    out, skip = [], False
    for i, tok in enumerate(head_run):
        if skip:
            skip = False
            continue
        if tok == "-e" and i + 1 < len(head_run) and head_run[i + 1].startswith(ALLOC + "="):
            skip = True
            continue
        out.append(tok)
    return out


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        h = Harness(Path(td))

        print("H1: default keeps stock allocator env on both ranks")
        _, head_a, worker_a = _launch(h)
        he_a, we_a = _env_of_head(head_a), _env_of_worker(worker_a)
        check(he_a.get(ALLOC) == STOCK, f"H1a head {ALLOC}={he_a.get(ALLOC)!r}")
        check(we_a.get(ALLOC) == STOCK, f"H1b worker {ALLOC}={we_a.get(ALLOC)!r}")
        inner_a = _inner_scripts(h)

        print("H2: TEST_DROP_ALLOC_CONF=1 + knob=0 drops env and nothing else")
        r = h.run(["eval", LAUNCH], entry="start.fn.sh",
                  GLM53_KV_OFFLOAD="0", GLM53_TEST_DROP_ALLOC_CONF="1")
        check(r.returncode == 0, f"H2a launch_cluster proceeds (rc={r.returncode})")
        calls = h.calls()
        head_b = next(c for c in calls if c[0] == "docker" and c[1] == "run" and "glm53-exl3-head" in c)
        worker_b = next(c[-1] for c in calls if c[0] == "ssh" and "docker run" in c[-1])
        he_b, we_b = _env_of_head(head_b), _env_of_worker(worker_b)
        check(ALLOC not in he_b, f"H2b head env dropped (got {he_b.get(ALLOC)!r})")
        check(ALLOC not in we_b, f"H2c worker env dropped (got {we_b.get(ALLOC)!r})")
        check(DROP_LOG in (r.stdout + r.stderr), "H2d drop announced in boot log")
        check(_strip_alloc_pair(head_a) == head_b,
              "H2e head docker-run argv exactly equal after removing only the alloc -e pair")
        exp_he = {k: v for k, v in he_a.items() if k != ALLOC}
        exp_we = {k: v for k, v in we_a.items() if k != ALLOC}
        check(he_b == exp_he, "H2f head env map = default minus only the alloc key")
        check(we_b == exp_we, "H2g worker env map = default minus only the alloc key")
        inner_b = _inner_scripts(h, GLM53_KV_OFFLOAD="0", GLM53_TEST_DROP_ALLOC_CONF="1")
        check(inner_b == inner_a, "H2h inner rank scripts byte-identical to the default boot")
        # implicit knob=0 (flag alone) behaves the same
        r = h.run(["eval", LAUNCH], entry="start.fn.sh", GLM53_TEST_DROP_ALLOC_CONF="1")
        check(r.returncode == 0 and DROP_LOG in (r.stdout + r.stderr), "H2i implicit knob=0 also drops")

        print("H3: fail-closed pre-stop refusals (rc=2, zero host calls)")
        r = h.run(["restart"], GLM53_KV_OFFLOAD="1", GLM53_TEST_DROP_ALLOC_CONF="1")
        check(r.returncode == 2 and not h.host_calls(),
              f"H3a knob=1 + flag=1 refused pre-stop (rc={r.returncode}, calls={len(h.host_calls())})")
        check("diagnostics-only flag for knob=0 boots" in (r.stderr + r.stdout), "H3b die names the rule")
        for junk in ("2", "yes", ""):
            r = h.run(["restart"], GLM53_TEST_DROP_ALLOC_CONF=junk)
            check(r.returncode == 2 and not h.host_calls()
                  and "GLM53_TEST_DROP_ALLOC_CONF" in (r.stderr + r.stdout),
                  f"H3c junk value {junk!r} refused pre-stop with named error (rc={r.returncode})")
        # in-launch_cluster belt-and-braces still fires if the guard is bypassed
        r = h.run(["eval", LAUNCH], entry="start.fn.sh",
                  GLM53_KV_OFFLOAD="1", GLM53_TEST_DROP_ALLOC_CONF="1")
        check(r.returncode != 0, f"H3d launch-site guard also refuses (rc={r.returncode})")

    print(f"\n{len(FAILURES)} failure(s)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
