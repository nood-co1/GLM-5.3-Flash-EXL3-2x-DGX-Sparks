#!/usr/bin/env bash
# GLM53_EXTRA_ENV: NAME=VALUE entries are appended to nccl_common as -e pairs; bad entries abort.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
blk="$(sed -n '/^    # Extra container env for diagnostics/,/^    fi$/p' "$HERE/../start.sh")"
[ -n "$blk" ] || { echo "extra-env block not found" >&2; exit 1; }
run() { # $1 = GLM53_EXTRA_ENV value; prints joined nccl_common or ERR
  bash -c "die() { echo \"DIE: \$*\" >&2; exit 2; }; log() { :; }; nccl_common=(); GLM53_EXTRA_ENV=\"$1\"; $(printf '%s' "$blk" | sed 's/^    //; s/^local _kv$//'); printf '%s ' \"\${nccl_common[@]}\"" 2>/dev/null || echo ERR
}
fail=0
check() { got="$(run "$1" | sed "s/[[:space:]]*$//")"; if [ "$got" = "$2" ]; then echo "ok   '$1'"; else echo "FAIL '$1' -> [$got] want [$2]"; fail=1; fi; }
check "" ""
check "VLLM_DEBUG_WORKSPACE=1" "-e VLLM_DEBUG_WORKSPACE=1"
check "A=1 B_2=x" "-e A=1 -e B_2=x"
check "bad-name=1" "ERR"
check "novalue" "ERR"
check "1ABC=2" "ERR"
exit $fail
