#!/bin/bash
# Move Pi Jarvis's code (both repos) to a tag, then restart when quiet.
# usage: jarvis-update <tag>    list tags: git -C ~/projects/Jarvis/backtalk tag -l 'pi-*'
# Rollback is the same command with the previous tag.
set -e
TAG=${1:?usage: jarvis-update <tag>}
J=$HOME/projects/Jarvis
for r in backtalk ai-visualizer; do
  git -C "$J/$r" fetch -q --tags origin
  git -C "$J/$r" checkout -q "$TAG"
  echo "$r: $(git -C "$J/$r" log --oneline -1)"
done
# pi-baseline predates pi/: keep the current venv, scripts and units then
if [ -f "$J/backtalk/pi/install.sh" ]; then "$J/backtalk/pi/install.sh" --user; fi
exec "$HOME/.local/bin/jarvis-restart-when-quiet" jarvis-face.service jarvis-voice.service
