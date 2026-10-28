#!/usr/bin/env bash
set -euo pipefail
usage() {
    echo "Usage: prepare.sh --inferno-root PATH"
    echo "Fetch pinned public inference dependencies, apply compatibility patches and create mountpoints."
    echo "INFERNO_ROOT may be used instead of --inferno-root."
}
here=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
fail() { echo "$*" >&2; exit 2; }
inferno=${INFERNO_ROOT:-}
while (($#)); do
    case "$1" in
        --inferno-root) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; inferno=$2; shift 2 ;;
        --help|-h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
[[ -n $inferno && -f $inferno/inferno/__init__.py ]] || {
    echo "Set --inferno-root to a fresh Inferno checkout." >&2; exit 2;
}
inferno=$(realpath "$inferno")
expected_inferno=36681d55ffb9b1b0e444bb1f980183df1e02074d
actual_inferno=$(git -C "$inferno" rev-parse HEAD)
[[ $actual_inferno == "$expected_inferno" ]] || {
    echo "Expected Inferno revision $expected_inferno, found $actual_inferno." >&2
    echo "In a fresh checkout, run: git -C \"$inferno\" checkout --detach $expected_inferno" >&2
    exit 2
}
for relative in inferno/sandbox/thunder assets data; do
    path=$inferno/$relative
    [[ ! -L $path ]] || {
        echo "Refusing symlink mountpoint: $path. Use a fresh checkout with real directories." >&2
        exit 2
    }
done
[[ ! -L $inferno/external ]] || fail "Refusing symlink dependency directory: $inferno/external"
while IFS='|' read -r relative url revision patch patch_hash; do
    [[ -n $relative && $relative != \#* ]] || continue
    target=$inferno/$relative
    [[ ! -L $target ]] || fail "Refusing symlink dependency directory: $target"
    if [[ ! -e $target/.git ]]; then
        if [[ -e $target ]]; then
            [[ -d $target && -z $(find "$target" -mindepth 1 -maxdepth 1 -print -quit) ]] ||
                fail "Dependency destination is not empty: $target. Use a fresh checkout."
        fi
        mkdir -p "$target"
        git -C "$target" init --quiet
        git -C "$target" remote add origin "$url"
    fi
    [[ $(git -C "$target" rev-parse --show-toplevel) == "$target" ]] ||
        fail "Dependency is not its own Git checkout: $target"
    origin=$(git -C "$target" remote get-url origin)
    [[ ${origin%.git} == "${url%.git}" ]] ||
        fail "Unexpected dependency origin in $target. Use a fresh checkout; existing source is unchanged."
    current=$(git -C "$target" rev-parse --verify HEAD 2>/dev/null || true)
    if [[ -z $current ]]; then
        # Exact public commits; do not follow branch tips or invoke recursive submodules.
        GIT_TERMINAL_PROMPT=0 git -C "$target" fetch --depth 1 origin "$revision"
        git -C "$target" checkout --detach "$revision"
    else
        [[ $current == "$revision" ]] ||
            fail "Unexpected dependency revision in $target: $current (expected $revision)."
    fi
    if [[ $patch != - ]]; then
        patch_file=$here/patches/$patch
        [[ -f $patch_file ]] || fail "Missing compatibility patch: $patch_file"
        actual_hash=$(sha256sum "$patch_file")
        [[ ${actual_hash%% *} == "$patch_hash" ]] || fail "Compatibility patch checksum mismatch: $patch"
        if git -C "$target" apply --reverse --check "$patch_file" 2>/dev/null; then
            echo "Compatibility patch already applied: $relative"
        elif git -C "$target" diff --quiet && git -C "$target" diff --cached --quiet &&
             git -C "$target" apply --check "$patch_file"; then
            git -C "$target" apply "$patch_file"
        else
            fail "Cannot safely apply compatibility patch in $target. Existing changes are unchanged."
        fi
    else
        git -C "$target" diff --quiet && git -C "$target" diff --cached --quiet ||
            fail "Unexpected tracked changes in $target."
    fi
    echo "$relative $revision"
done < "$here/dependencies.lock"
mkdir -p "$inferno/inferno/sandbox/thunder" "$inferno/assets" "$inferno/data"
