#!/bin/bash
# jarvis-voice.service: the voice line in tmux, so the live session can be
# typed into: tmux -L voice attach (detach with Ctrl-b d, never Ctrl-c).
# xvfb-run: pynput's talk-key listener needs an X display.
VAULT="$HOME/projects/Jarvis/Jarvis Memory"
for i in $(seq 60); do mountpoint -q "$VAULT" && break; sleep 1; done
mountpoint -q "$VAULT" || echo "vault not mounted after 60 s; starting anyway" >&2
cd "$HOME/projects/Jarvis/backtalk" || exit 1
tmux -L voice kill-server 2>/dev/null
tmux -L voice new-session -d -s voice "xvfb-run -a .venv/bin/python -m backtalk.main"
while tmux -L voice has-session -t voice 2>/dev/null; do sleep 2; done
echo "voice line exited" >&2
exit 1
