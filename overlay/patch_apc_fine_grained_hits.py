#!/usr/bin/env python3
"""Re-enable fine-grained (hash-block) prefix-cache hits at a kpool-safe alignment.

Problem
-------
``HybridKVCacheCoordinator.__init__`` computes ``enable_partial_hash_hits``, then
vetoes it if ANY single-type manager reports
``supports_fine_grained_hash_lookup == False`` while its ``block_size`` differs
from ``hash_block_size``.  The scan covers **every** manager, including groups
whose spec sets ``participates_in_prefix_caching = False``.

On this kit that is ``KpoolTailManager`` (``KpoolTailSpec.block_size ==
index_kpool == 4``), so the boot log says::

    WARNING [kv_cache_coordinator.py:635] Disabling fine-grained prefix-cache
    hits because these KV cache managers require block-aligned lookups:
    KpoolTailManager.

The veto is spurious.  ``verify_and_split_kv_cache_groups()`` already skips
non-participating groups, so ``KpoolTailManager.find_longest_cache_hit`` is
never called by the coordinator and its
``supports_fine_grained_hash_lookup`` flag cannot affect any lookup.  The cost
is real: ``_cache_hit_alignment_tokens`` falls back to
``scheduler_block_size`` (3584 here) instead of ``hash_block_size`` (64), so
every warm turn re-prefills up to 3583 already-computed tokens (~1.5-3 s).

What a non-participating scratch group *does* require
-----------------------------------------------------
Mia's rule -- "wrong indexer tail state is fatal" -- is about state, not
lookups.  ``KpoolTailSpec`` is a one-block circular buffer holding the
in-progress (incomplete) kpool's raw K + gate score, addressed by
``pos % kpool``.  A warm hit ending at ``H`` allocates that block fresh and
prefills only ``[H, N)``, so the ``H % kpool`` raw entries belonging to the
current in-progress pool are never recomputed.  With
``index_kpool_always_select_tail = true`` the indexer then compresses garbage.

So the real invariant is ``H % kpool == 0``, i.e. the hit alignment must be a
multiple of every non-participating group's ``block_size``.  Here
``hash_block_size (64) % index_kpool (4) == 0``, so 64 is already kpool-safe
and the required alignment is ``lcm(64, 4) == 64`` -- no change to
``_cache_hit_alignment_tokens`` itself is needed or wanted (the fine-grained
lookup paths index the raw hash list positionally and are only sound when
``alignment_tokens == hash_block_size``; see docs/DESIGN-apc-fine-grained-hits.md §2).

This patch therefore
--------------------
1. scopes the ``supports_fine_grained_hash_lookup`` check to groups that
   actually participate in prefix caching; and
2. replaces the accidental veto with the invariant a scratch group really
   needs, **verified at runtime from the actual specs**:
   ``hash_block_size % <scratch alignment> == 0``.

Fail-closed, in both directions
-------------------------------
* A *participating* manager that cannot answer a fine lookup is upstream's own
  condition: fine hits are DISABLED and we log upstream's warning.  That is the
  safe, correct fallback and it is what upstream does.
* A *non-participating scratch* group whose alignment requirement cannot be
  verified, or is verified and violated, is a different animal: the safety
  argument for this patch does not hold on that layout, and silently degrading
  to block-aligned hits would hide the fact.  The coordinator **refuses to
  start**, raising ``Glm53FineGrainedAPCError`` with a ``[glm53-apc-finegrained]``
  message that names the group, the alignment, and the remedy.
  ``GLM53_FINEGRAINED_APC=0`` is the documented escape hatch: it restores the
  upstream (all-managers) veto verbatim and never raises.
* Manager/group cardinality is asserted before iterating -- upstream's ``zip()``
  would silently truncate the scan if the two lists ever diverged.

Patcher fail-closed / transactional
-----------------------------------
* Anchors are counted before anything is mutated; drift aborts with no write.
* The new text is compiled and re-validated **before** it is written.
* The write is atomic (temp file in the same directory + ``os.replace``), so an
  interrupted run can never leave a half-patched coordinator.
* If ``MARK`` is already present the patcher does not just skip: it validates
  the complete patched state (helper block, gate, kill switch, tag, no
  surviving upstream veto) and fails closed if anything is missing.

Idempotent, MARK/anchor guarded.  Order-independent with respect to
``patch_hybrid_prefix_hit.py`` (different anchors; shared helper insert point is
guarded by name).

Kill switch: ``GLM53_FINEGRAINED_APC=0`` in the engine environment restores the
upstream (all-managers) veto at runtime without unpatching.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

P = Path(
    os.environ.get(
        "GLM53_KV_COORDINATOR_PY",
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py",
    )
)
MARK = "# [glm53-finegrained-apc]"
# Runtime message tag for the refusals raised at coordinator init.
RUNTIME_TAG = "[glm53-apc-finegrained]"
HELPER_BEGIN = "# [glm53-finegrained-apc] helper-begin"
HELPER_END = "# [glm53-finegrained-apc] helper-end"
HELPER_NEEDLE = "def _validate_prefix_cache_retention_interval(\n"
IMPORT_ANCHOR = "from abc import ABC, abstractmethod\n"

# Where the helper block goes. ``patch_hybrid_prefix_hit.py`` inserts its own
# helper before the same upstream needle, so "insert before the needle" makes
# the composed file depend on which patch ran first. Anchoring ahead of a
# sibling glm53 helper when one is already present makes the two patches
# byte-commutative in either order (host test A4).
HELPER_ANCHORS = (
    "\ndef _glm53_inner_kv_spec(spec):\n",  # patch_hybrid_prefix_hit.py
    HELPER_NEEDLE,
)


def helper_anchor(text: str) -> str | None:
    for anchor in HELPER_ANCHORS:
        if anchor in text:
            return anchor
    return None

HELPER = '''
# [glm53-finegrained-apc] helper-begin
GLM53_FG_TAG = "[glm53-apc-finegrained]"


class Glm53FineGrainedAPCError(RuntimeError):
    """A fine-grained-hit invariant could not be verified, or was violated.

    Raised from ``HybridKVCacheCoordinator.__init__`` so the engine refuses to
    start rather than silently serving a layout whose safety argument does not
    hold.  ``GLM53_FINEGRAINED_APC=0`` restores the upstream gate.
    """


def _glm53_scratch_alignment(manager, spec):
    """Token alignment a non-participating (scratch) group requires.

    Read from the ACTUAL manager/spec objects -- never assumed, never
    defaulted.  Returns ``(alignment, source)``, or ``(None, reason)`` when the
    group cannot be verified, in which case the caller must refuse to start.

    Resolution order:

    1. An explicit capability, ``fine_grained_hit_alignment`` on the manager or
       the spec.  This is the forward-compatible hook: a scratch manager that
       knows its own requirement states it, and nothing has to be inferred.
    2. Otherwise ``spec.block_size``, cross-checked against
       ``manager.block_size`` (they must agree) and, when the spec exposes it,
       against ``spec.index_kpool`` / ``spec.kpool``.  For GLM5Next's
       ``KpoolTailSpec`` this is exactly ``index_kpool``, which is the quantity
       the invariant is really about.

    Anything unverifiable -- a missing block_size, a non-integer, or two
    sources that disagree -- returns ``None`` and is treated as a hard failure.
    """
    for owner, owner_name in ((manager, "manager"), (spec, "spec")):
        declared = getattr(owner, "fine_grained_hit_alignment", None)
        if declared is not None:
            try:
                declared_int = int(declared)
            except (TypeError, ValueError):
                return None, (
                    f"{owner_name}.fine_grained_hit_alignment={declared!r} "
                    "is not an integer"
                )
            return declared_int, f"{owner_name}.fine_grained_hit_alignment"

    spec_bs = getattr(spec, "block_size", None)
    mgr_bs = getattr(manager, "block_size", None)
    if spec_bs is None:
        return None, (
            f"{type(spec).__name__} exposes neither block_size nor "
            "fine_grained_hit_alignment"
        )
    try:
        spec_bs = int(spec_bs)
    except (TypeError, ValueError):
        return None, f"{type(spec).__name__}.block_size={spec_bs!r} is not an integer"
    if mgr_bs is None:
        return None, f"{type(manager).__name__} exposes no block_size to cross-check"
    try:
        mgr_bs = int(mgr_bs)
    except (TypeError, ValueError):
        return None, f"{type(manager).__name__}.block_size={mgr_bs!r} is not an integer"
    if spec_bs != mgr_bs:
        return None, (
            f"spec.block_size={spec_bs} disagrees with "
            f"manager.block_size={mgr_bs}"
        )

    source = "spec.block_size"
    for attr in ("index_kpool", "kpool"):
        declared = getattr(spec, attr, None)
        if declared is None:
            continue
        try:
            declared_int = int(declared)
        except (TypeError, ValueError):
            return None, f"{type(spec).__name__}.{attr}={declared!r} is not an integer"
        if declared_int != spec_bs:
            return None, (
                f"{type(spec).__name__}.{attr}={declared_int} disagrees with "
                f"block_size={spec_bs}"
            )
        source = f"spec.block_size == spec.{attr}"
    return spec_bs, source


def _glm53_connector_receipt():
    """Boot receipt: is a KV-transfer connector configured?

    ``kv_cache_manager.truncate_computed_blocks`` asserts
    ``num_computed_tokens % manager.block_size == 0``, which a 64-aligned hit
    violates for a 3584-block MLA manager.  Its only caller is gated on
    ``connector is not None and connector.supports_divergent_local_hybrid_hits``,
    so the assert is unreachable with no ``--kv-transfer-config``.  This line
    makes that precondition an explicit boot receipt instead of an assumption
    (see DESIGN R1).  Best effort only: it is a log string, never a gate, and
    must never be able to break engine init.
    """
    try:
        from vllm.config import get_current_vllm_config

        cfg = get_current_vllm_config()
        kv_transfer = getattr(cfg, "kv_transfer_config", None)
        if kv_transfer is None:
            return "absent (no kv_transfer_config) -- truncate_computed_blocks unreachable"
        return (
            "PRESENT ("
            + repr(getattr(kv_transfer, "kv_connector", kv_transfer))
            + ") -- see DESIGN R1: truncate_computed_blocks may assert"
        )
    except Exception as exc:  # pragma: no cover - diagnostics only
        return f"unknown ({type(exc).__name__})"


def _glm53_finegrained_hit_gate(managers, kv_cache_groups, hash_block_size):
    """Decide whether fine-grained (hash-block-aligned) hits are state-safe.

    Returns ``(enable, blockers, scratch)``; raises
    ``Glm53FineGrainedAPCError`` when an invariant cannot be verified or is
    violated.

    Two distinct populations, two distinct questions:

    * Groups with ``participates_in_prefix_caching = True`` are in
      ``attention_groups`` and DO run ``find_longest_cache_hit``.  They must be
      able to answer a lookup at ``hash_block_size`` granularity: either the
      manager advertises ``supports_fine_grained_hash_lookup``, or its
      ``block_size`` already equals ``hash_block_size`` (nothing finer is
      asked of it).  This is the upstream check, correctly scoped, and it
      DISABLES fine hits rather than raising -- block-aligned hits are the
      correct, safe fallback for that case.

    * Groups with ``participates_in_prefix_caching = False`` (GLM5Next's
      ``KpoolTailSpec``) are skipped by ``verify_and_split_kv_cache_groups``
      and never looked up, so their lookup capability is irrelevant.  What
      they need is that the hit lands where their per-request state is EMPTY.
      The kpool indexer tail holds an in-progress pool of ``index_kpool``
      tokens addressed by ``pos % kpool``; a hit at ``H`` leaves
      ``H % kpool`` raw K/gate entries unrecomputed, which
      ``index_kpool_always_select_tail`` would then compress.  Require
      ``hash_block_size % alignment == 0`` so every reachable hit boundary
      lands on an empty pool -- and RAISE if it does not hold, or cannot be
      checked.  Degrading silently here would hide the fact that this patch's
      safety argument does not apply to the running layout.

    Cardinality is asserted before iterating: upstream pairs these two lists
    with ``zip()``, which would silently truncate the scan -- and therefore
    skip real blockers -- if they ever diverged.
    """
    managers = list(managers)
    groups = list(kv_cache_groups)
    if len(managers) != len(groups):
        raise Glm53FineGrainedAPCError(
            f"{GLM53_FG_TAG} manager/group cardinality mismatch: "
            f"{len(managers)} single-type managers vs {len(groups)} KV cache "
            "groups. zip() would silently truncate the fine-grained-hit "
            "validation and skip real blockers, so this layout cannot be "
            "validated. Refusing to start. Set GLM53_FINEGRAINED_APC=0 to "
            "restore the upstream (all-managers, block-aligned) gate."
        )
    if not managers:
        raise Glm53FineGrainedAPCError(
            f"{GLM53_FG_TAG} no KV cache managers to validate; refusing to "
            "enable fine-grained prefix-cache hits. Set "
            "GLM53_FINEGRAINED_APC=0 to restore the upstream gate."
        )
    if not isinstance(hash_block_size, int) or hash_block_size <= 0:
        raise Glm53FineGrainedAPCError(
            f"{GLM53_FG_TAG} hash_block_size={hash_block_size!r} is not a "
            "positive integer; the fine-grained hit alignment is undefined. "
            "Refusing to start. Set GLM53_FINEGRAINED_APC=0 to restore the "
            "upstream gate."
        )

    blockers: list[str] = []
    scratch: dict[str, int] = {}
    for index, (manager, group) in enumerate(zip(managers, groups)):
        name = type(manager).__name__
        spec = getattr(group, "kv_cache_spec", None)
        if spec is None:
            raise Glm53FineGrainedAPCError(
                f"{GLM53_FG_TAG} KV cache group {index} (manager {name}) has "
                "no kv_cache_spec; the fine-grained-hit invariants cannot be "
                "verified. Refusing to start. Set GLM53_FINEGRAINED_APC=0 to "
                "restore the upstream gate."
            )
        if getattr(spec, "participates_in_prefix_caching", True):
            block_size = getattr(manager, "block_size", None)
            supports_fine = getattr(
                manager, "supports_fine_grained_hash_lookup", False
            )
            if not supports_fine and block_size != hash_block_size:
                blockers.append(
                    f"{name}"
                    f"(participating, block_size={block_size}, "
                    f"block-aligned lookups only)"
                )
            continue

        alignment, source = _glm53_scratch_alignment(manager, spec)
        if alignment is None:
            raise Glm53FineGrainedAPCError(
                f"{GLM53_FG_TAG} cannot verify the hit-alignment requirement "
                f"of non-participating scratch group {index} {name} "
                f"({type(spec).__name__}): {source}. A fine-grained hit could "
                "resume mid-pool and leave un-recomputed raw K/gate entries. "
                "Refusing to start. Set GLM53_FINEGRAINED_APC=0 to fall back "
                "to block-aligned (scheduler_block_size) hits."
            )
        scratch[name] = alignment
        if alignment <= 0 or hash_block_size % alignment != 0:
            raise Glm53FineGrainedAPCError(
                f"{GLM53_FG_TAG} scratch group {index} {name} "
                f"({type(spec).__name__}) requires hit alignment "
                f"{alignment} (verified from {source}), but "
                f"hash_block_size={hash_block_size} is not a multiple of it. "
                "Every reachable fine-grained hit boundary H would satisfy "
                f"H % {alignment} != 0 for some H, resuming mid-pool and "
                "leaving un-recomputed raw K/gate entries that "
                "index_kpool_always_select_tail would then compress. "
                "Refusing to start. Set GLM53_FINEGRAINED_APC=0 to fall back "
                "to block-aligned (scheduler_block_size) hits."
            )
    return (not blockers), blockers, scratch


# [glm53-finegrained-apc] helper-end


'''

GATE_OLD = """        if self.enable_partial_hash_hits:
            unsupported_partial_hit_managers = {
                type(manager).__name__
                for manager in self.single_type_managers
                if not manager.supports_fine_grained_hash_lookup
                and manager.block_size != hash_block_size
            }
            if unsupported_partial_hit_managers:
                self.enable_partial_hash_hits = False
                logger.warning_once(
                    "Disabling fine-grained prefix-cache hits because these KV "
                    "cache managers require block-aligned lookups: %s.",
                    ", ".join(sorted(unsupported_partial_hit_managers)),
                )
"""

GATE_NEW = """        if self.enable_partial_hash_hits:
            # [glm53-finegrained-apc] Upstream scans EVERY manager here,
            # including groups whose spec sets
            # participates_in_prefix_caching=False. Those groups are already
            # skipped by verify_and_split_kv_cache_groups(), so their
            # supports_fine_grained_hash_lookup flag can never affect a
            # lookup -- but vetoing on it silently pins every hit to
            # scheduler_block_size. GLM5Next's KpoolTailManager
            # (block_size == index_kpool == 4) does exactly that, costing up
            # to scheduler_block_size-1 recomputed tokens per warm turn.
            # Scope the flag check to participating groups and enforce the
            # invariant a scratch group actually needs instead: the hit must
            # land where its per-request state is empty
            # (hash_block_size % alignment == 0), verified at runtime from the
            # actual specs. A scratch group that fails or cannot be verified
            # RAISES -- see _glm53_finegrained_hit_gate.
            if os.environ.get("GLM53_FINEGRAINED_APC", "1") == "0":
                _glm53_ok = False
                _glm53_blockers = ["GLM53_FINEGRAINED_APC=0 (kill switch)"]
                _glm53_scratch: dict = {}
            else:
                _glm53_ok, _glm53_blockers, _glm53_scratch = (
                    _glm53_finegrained_hit_gate(
                        self.single_type_managers,
                        kv_cache_config.kv_cache_groups,
                        hash_block_size,
                    )
                )
            self.enable_partial_hash_hits = _glm53_ok
            if _glm53_ok:
                logger.info(  # [glm53-finegrained-apc]
                    "Fine-grained prefix-cache hits ENABLED: alignment=%d "
                    "tokens (hash_block_size), was scheduler_block_size=%d. "
                    "Verified non-participating scratch alignments %s all "
                    "divide the alignment, so every reachable hit boundary "
                    "leaves their per-request state empty. "
                    "KV-transfer connector: %s.",
                    hash_block_size,
                    self.scheduler_block_size,
                    _glm53_scratch or {},
                    _glm53_connector_receipt(),
                )
            else:
                logger.warning_once(
                    "Disabling fine-grained prefix-cache hits because these KV "
                    "cache managers require block-aligned lookups: %s.",
                    ", ".join(sorted(_glm53_blockers)),
                )
"""

# Every token that must be present in a correctly patched file. Used both to
# validate what we are about to write and to validate a file that already
# carries MARK (a pre-existing marker must never be trusted on its own).
REQUIRED_AFTER = (
    MARK,
    HELPER_BEGIN,
    HELPER_END,
    RUNTIME_TAG,
    "class Glm53FineGrainedAPCError(RuntimeError):",
    "def _glm53_scratch_alignment(",
    "def _glm53_connector_receipt(",
    "def _glm53_finegrained_hit_gate(",
    "GLM53_FINEGRAINED_APC",
    "Fine-grained prefix-cache hits ENABLED",
    "manager/group cardinality mismatch",
)

UNIQUE_AFTER = (
    HELPER_BEGIN,
    HELPER_END,
    "def _glm53_scratch_alignment(",
    "def _glm53_connector_receipt(",
    "def _glm53_finegrained_hit_gate(",
)


def patched_problems(text: str) -> list[str]:
    """Everything wrong with a file that claims to be patched."""
    problems: list[str] = []
    for token in REQUIRED_AFTER:
        if token not in text:
            problems.append(f"missing {token!r}")
    for token in UNIQUE_AFTER:
        n = text.count(token)
        if n != 1:
            problems.append(f"{token!r} appears {n} times (expected 1)")
    if "unsupported_partial_hit_managers" in text:
        problems.append("upstream all-managers veto still present")
    if "\nimport os\n" not in text:
        problems.append("missing 'import os'")
    try:
        compile(text, str(P), "exec")
    except SyntaxError as exc:
        problems.append(f"does not compile: {exc}")
    return problems


def pristine_problems(text: str) -> list[str]:
    """Anchor drift detection, run before anything is mutated."""
    problems: list[str] = []
    n = text.count(GATE_OLD)
    if n != 1:
        problems.append(f"expected one partial-hit-gate target, found {n}")
    n = text.count(HELPER_NEEDLE)
    if n != 1:
        problems.append(f"expected one helper insert point, found {n}")
    anchor = helper_anchor(text)
    if anchor is None:
        problems.append("no helper insert anchor found")
    elif text.count(anchor) != 1:
        problems.append(
            f"helper insert anchor {anchor!r} is not unique "
            f"(found {text.count(anchor)})"
        )
    if "\nimport os\n" not in text and text.count(IMPORT_ANCHOR) != 1:
        problems.append(
            f"import anchor not unique (found {text.count(IMPORT_ANCHOR)})"
        )
    for stray in (
        HELPER_BEGIN,
        HELPER_END,
        "def _glm53_scratch_alignment(",
        "def _glm53_connector_receipt(",
        "def _glm53_finegrained_hit_gate(",
        "class Glm53FineGrainedAPCError(",
    ):
        if stray in text:
            problems.append(
                f"{stray!r} already present without {MARK} "
                "(partial or foreign patch state)"
            )
    return problems


def atomic_write(path: Path, text: str) -> None:
    """Write via a temp file in the same directory + os.replace.

    An interrupted or failing run must never leave a half-patched coordinator:
    either the old bytes or the new bytes, never a mixture.
    """
    tmp = path.with_name(path.name + ".glm53-finegrained.tmp")
    try:
        tmp.write_text(text)
        try:
            os.chmod(tmp, os.stat(path).st_mode & 0o7777)
        except OSError:
            pass
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def main() -> int:
    if not P.is_file():
        raise SystemExit(f"missing {P}")
    text = P.read_text()

    # A pre-existing marker is not proof of a complete patch: validate it.
    if MARK in text:
        problems = patched_problems(text)
        if problems:
            raise SystemExit(
                f"{P}: {MARK} is present but the patch is INCOMPLETE - "
                "refusing to leave a half-patched coordinator in place: "
                + "; ".join(problems)
            )
        print(f"{P.name}: {MARK} already present and complete - skipping")
        return 0

    problems = pristine_problems(text)
    if problems:
        raise SystemExit(f"{P}: " + "; ".join(problems))

    # The patched gate reads os.environ; upstream does not import os here
    # (checked against the live file: only abc/collections/typing + vllm).
    if "\nimport os\n" not in text:
        text = text.replace(IMPORT_ANCHOR, "import os\n" + IMPORT_ANCHOR, 1)

    anchor = helper_anchor(text)
    text = text.replace(anchor, HELPER + anchor, 1)
    text = text.replace(GATE_OLD, GATE_NEW, 1)

    # Validate the full result BEFORE it touches the filesystem.
    problems = patched_problems(text)
    if problems:
        raise SystemExit(
            f"{P}: refusing to write an invalid patched file: "
            + "; ".join(problems)
        )

    atomic_write(P, text)

    # And validate what actually landed on disk.
    problems = patched_problems(P.read_text())
    if problems:
        raise SystemExit(
            f"{P}: post-write validation failed: " + "; ".join(problems)
        )

    print(
        f"patched {P.name} (fine-grained APC: veto scoped to participating "
        f"groups; scratch groups verified for alignment divisibility, "
        f"refusing to start if violated)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
