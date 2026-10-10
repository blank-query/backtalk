#!/bin/bash
# Pi Jarvis on bare metal (Raspberry Pi OS bookworm, the desktop user). Idempotent.
# Layout: code in ~/projects/Jarvis/{backtalk,ai-visualizer} (forks, tags pi-*),
# state in ~/projects/Jarvis (CLAUDE.md, .backtalk), ~/.claude, ~/jarvis-data
# (signals, models, voices, plugins); secrets in ~/.config/jarvis/jarvis.env (600).
# usage: JARVIS_VAULT_SRC=user@nas:/vault/path pi/install.sh
#                               full install (sudo: apt, sysctl, linger, vault unit)
#        pi/install.sh --user   venv, scripts and user units only (what jarvis-update runs)
set -e
B=$(cd "$(dirname "$0")/.." && pwd)
export PATH="$HOME/.local/bin:$PATH"

if [ "$1" != --user ]; then
  sudo apt-get update -q
  sudo apt-get install -y --no-install-recommends tmux xvfb xauth espeak-ng \
    libportaudio2 libsndfile1 build-essential iputils-arping sshfs sshpass curl
  # this host may carry a second address on eth0: answer ARP only for an
  # address on the asking interface, so Wi-Fi (if ever on) can't claim it.
  printf 'net.ipv4.conf.all.arp_ignore=1\nnet.ipv4.conf.all.arp_announce=2\n' \
    | sudo tee /etc/sysctl.d/90-jarvis-arp.conf >/dev/null
  sudo /usr/sbin/sysctl -q -p /etc/sysctl.d/90-jarvis-arp.conf
  sudo loginctl enable-linger "$USER"
  # vault from the home server; password file /etc/jarvis-vault.pass (root, 600) is set by hand
  V="$HOME/projects/Jarvis/Jarvis Memory"
  if ! mountpoint -q "$V"; then   # unmounted it stays a read-only stub, so nothing writes a stray vault
    mkdir -p "$(dirname "$V")" && sudo mkdir -p "$V"
    echo "# THE VAULT IS NOT MOUNTED (jarvis-vault-home.service on this host)" | sudo tee "$V/VAULT-INDEX.md" >/dev/null
    sudo chown -R root:root "$V"
  fi
  SRC=${JARVIS_VAULT_SRC:?set JARVIS_VAULT_SRC=user@nas:/path/to/vault}
  sed -e "s|@HOME@|$HOME|" -e "s|@VAULT_SRC@|$SRC|" "$B/pi/systemd/jarvis-vault-home.service" \
    | sudo tee /etc/systemd/system/jarvis-vault-home.service >/dev/null
  sudo systemctl daemon-reload && sudo systemctl enable --now jarvis-vault-home.service
fi

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
cd "$B"
[ -x .venv/bin/python ] || uv venv -q --python 3.11.17 .venv
uv pip sync -q --python .venv/bin/python pi/requirements.txt
mkdir -p ~/.local/bin ~/.config/systemd/user ~/jarvis-data/signals logs
ln -sf "$B/.venv/lib/python3.11/site-packages/claude_agent_sdk/_bundled/claude" ~/.local/bin/claude
# the scripts live outside the checkout, so checking out an older tag can't remove them
install -m 755 pi/run-voice.sh ~/.local/bin/jarvis-run-voice
install -m 755 pi/update.sh ~/.local/bin/jarvis-update
install -m 755 pi/restart-when-quiet.sh ~/.local/bin/jarvis-restart-when-quiet
install -m 644 pi/systemd/jarvis-face.service pi/systemd/jarvis-voice.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable jarvis-face.service jarvis-voice.service
echo "installed. start: systemctl --user start jarvis-face jarvis-voice"
