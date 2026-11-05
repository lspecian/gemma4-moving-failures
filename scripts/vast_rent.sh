#!/usr/bin/env bash
# Rent the first offer that actually starts: try the cheapest matching offers in turn; an instance created with
# cur_state=stopped (its GPU was taken between search and create: stale listings on popular GPUs, 2026-10-08 x4)
# is destroyed at once and the next offer is tried. Prints the instance id and offer id of the one that runs.
#
#   scripts/vast_rent.sh '<search query>' <max $/h> [tries]
#   e.g. scripts/vast_rent.sh 'gpu_name in [H100_SXM,H100_NVL] num_gpus=1 reliability>0.98 inet_down>300 disk_space>120 rentable=true' 3.2
set -u
Q=$1; MAX=$2; TRIES=${3:-6}
for OFFER in $(vastai search offers "$Q" -o 'dph' --raw 2>/dev/null | python3 -c "
import json,sys
for o in json.load(sys.stdin):
    if o['dph_total'] <= float(sys.argv[1]): print(o['id'])" "$MAX" | head -"$TRIES"); do
  INST=$(vastai create instance "$OFFER" --image vllm/vllm-openai:v0.19.1 --disk 120 --ssh --direct \
    --label gemma4-vllm-eval --raw 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin).get("new_contract") or "")')
  [ -z "$INST" ] && { echo "offer $OFFER: create failed" >&2; continue; }
  for i in 1 2 3 4 5 6; do
    st=$(vastai show instance "$INST" --raw 2>/dev/null | python3 -c "import json,sys; i=json.load(sys.stdin); print(i.get('cur_state'), i.get('intended_status'))")
    case "$st" in *stopped*) break;; running*) break;; esac; sleep 5
  done
  case "$st" in
    *stopped*) echo "offer $OFFER -> instance $INST created stopped; destroying" >&2; vastai destroy instance -y "$INST" >/dev/null 2>&1;;
    *) echo "$INST $OFFER"; exit 0;;
  esac
done
echo "no offer started" >&2; exit 1
