#!/usr/bin/env bash
# Export one version of the agent into a directory the harness can run.
#
#   scripts/materialize.sh <version>[+<overlay>...] <out_dir> [--overlay <name>]...
#
# <version> is one of:
#   WORKTREE              the current, possibly uncommitted, agent/ directory
#   <git ref>             agent/ at that ref, e.g. agent-v5, HEAD~1, c432a10
#   <git ref>:<path>      any directory at that ref, e.g. c432a10:submissions/v3 (pre-restructure versions)
#   baselines/starter     a plain directory in the working tree
# Overlays are experiments/variants/<name>/ in the working tree, applied in the order given after the export:
# every file of the variant is copied over the exported agent, except files whose name starts with "_" at the
# variant root (overlay metadata such as _README.md). If the variant has _post.sh, it then runs as
# `bash _post.sh <out_dir>` (for changes that are not plain file copies, e.g. splitting a skill).
# `WORKTREE+pipeline+think768` is shorthand for `WORKTREE --overlay pipeline --overlay think768`, so
# scripts/run_ab.sh can take overlaid versions unchanged.
# Writes <out_dir>.version with the resolved ref, commit and overlays, so results stay traceable.
set -euo pipefail
VER=$1; OUT=$2; shift 2
OVERLAYS=()
if [[ "$VER" == *+* ]]; then
  IFS=+ read -r -a PARTS <<< "$VER"
  VER=${PARTS[0]}
  OVERLAYS+=("${PARTS[@]:1}")
fi
while (( $# )); do
  case $1 in
    --overlay) OVERLAYS+=("$2"); shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
for ov in ${OVERLAYS[@]+"${OVERLAYS[@]}"}; do
  [[ -d "experiments/variants/$ov" ]] || { echo "no such overlay: experiments/variants/$ov" >&2; exit 2; }
done
rm -rf "$OUT" && mkdir -p "$OUT"

if [[ "$VER" == WORKTREE ]]; then
  cp -R agent/. "$OUT/"
  echo "WORKTREE $(git rev-parse --short HEAD)$(git diff --quiet HEAD -- agent || echo '+dirty')" > "$OUT/VERSION"
elif [[ -d "$VER" ]]; then
  cp -R "$VER/." "$OUT/"
  echo "$VER $(git rev-parse --short HEAD)" > "$OUT/VERSION"
else
  REF=${VER%%:*}; P=agent
  [[ "$VER" == *:* ]] && P=${VER#*:}
  git archive "$REF" "$P" | tar -x -C "$OUT" --strip-components=$(awk -F/ '{print NF}' <<< "$P")
  echo "$VER $(git rev-parse --short "$REF")" > "$OUT/VERSION"
fi

for ov in ${OVERLAYS[@]+"${OVERLAYS[@]}"}; do
  SRC="experiments/variants/$ov"
  while IFS= read -r f; do
    mkdir -p "$OUT/$(dirname "$f")"
    cp "$SRC/$f" "$OUT/$f"
  done < <(cd "$SRC" && find . -type f ! -path './_*' ! -name .DS_Store ! -path '*/__pycache__/*' | sort)
  if [[ -f "$SRC/_post.sh" ]]; then
    bash "$SRC/_post.sh" "$OUT"
  fi
  DIRTY=""
  git diff --quiet HEAD -- "$SRC" || DIRTY="+dirty"
  [[ -z "$(git ls-files --others --exclude-standard -- "$SRC")" ]] || DIRTY="+dirty"
  echo "overlay $ov $(git rev-parse --short HEAD)$DIRTY" >> "$OUT/VERSION"
done

find "$OUT" -name __pycache__ -type d -prune -exec rm -rf {} +
find "$OUT" -name .DS_Store -delete
# VERSION is not an allowed submission file type; keep it next to, not inside, what gets zipped.
mv "$OUT/VERSION" "$OUT.version"
echo "$OUT <- $(tr '\n' ' ' < "$OUT.version")"
