#!/usr/bin/env bash
# Render startup script — restores the agent's Ed25519 signing key from env var,
# then launches the ConnectOnion agent.
#
# Required env var (set in Render dashboard):
#   CO_AGENT_KEY_B64 — base64-encoded contents of .co/keys/agent.key
#   Get it locally with: base64 -i .co/keys/agent.key | tr -d '\n'

set -e

# Restore .co/keys/agent.key from the env var
if [ -n "$CO_AGENT_KEY_B64" ]; then
    mkdir -p .co/keys
    echo "$CO_AGENT_KEY_B64" | tr -d '\n\r ' | base64 --decode > .co/keys/agent.key
    chmod 600 .co/keys/agent.key
    echo "[start.sh] Agent key restored from CO_AGENT_KEY_B64"
else
    echo "[start.sh] WARNING: CO_AGENT_KEY_B64 not set — agent will generate a new address"
fi

# Strip any trailing whitespace/newlines from OPENONION_API_KEY before the
# Python process starts — Render dashboard pastes sometimes add a trailing \n
# which causes requests to throw InvalidHeader.
if [ -n "$OPENONION_API_KEY" ]; then
    OPENONION_API_KEY=$(printf '%s' "$OPENONION_API_KEY" | tr -d '\n\r')
    export OPENONION_API_KEY
fi

exec python agent.py
