#!/usr/bin/env bash
# Tear down ONE rented GPU: destroy the instance, stop only its own vast_up.sh loop and its own SSH tunnel.
# (2026-10-07: a chain's cleanup killed every `ssh -N` / vast_up.sh on the Mac, which cut the tunnel of a second
# instance mid-run and lost ~25 tasks per arm. Never kill tunnels by a generic pattern.)
#
#   scripts/vast_down.sh <instance_id> <local_port> [offer_id]
#     local_port: the LOCAL_PORT the instance's vast_up.sh used (default 8002); offer_id: the offer vast_up.sh was
#     started with (its loop is matched by "vast_up.sh <offer_id>"; omit to leave the loop, which exits on its own
#     only when killed).
set -u
INST=$1; LPORT=${2:-8002}; OFFER=${3:-}
vastai destroy instance -y "$INST" 2>&1 | tail -1
if [ -n "$OFFER" ]; then
  ps -axo pid,command | grep -v grep | grep -E "vast_up\.sh $OFFER( |$)" | awk '{print $1}' | xargs -r kill
fi
ps -axo pid,command | grep -v grep | grep -E "ssh -N .*-L $LPORT:127\.0\.0\.1:8000" | awk '{print $1}' | xargs -r kill
echo "torn down instance $INST (tunnel :$LPORT${OFFER:+, loop for offer $OFFER})"
