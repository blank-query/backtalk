#!/bin/bash
# Restart Pi Jarvis only once nobody's talking: the voice idle AND no open
# mic mid-utterance, continuously for 5 s. Sir's explicit "go" for the
# restart still comes first; this only picks the moment.
# usage: jarvis-restart-when-quiet [unit ...]   (default jarvis-voice.service)
# Runs in its own transient unit, so the caller returns at once and the wait
# survives the restart. Output: journalctl --user -u 'jarvis-restart-*'.
if [ -z "$JARVIS_RQ_UNIT" ]; then
    systemd-run --user --collect --quiet --unit="jarvis-restart-$(date +%s)" \
        --setenv=JARVIS_RQ_UNIT=1 "$(realpath "$0")" "$@" \
        && echo "restart queued: waits for 5 s of quiet (up to 10 min), then restarts ${*:-jarvis-voice.service}"
    exit
fi
BUS=$HOME/jarvis-data/signals
quiet=0; capped=0
for i in $(seq 1 1200); do                       # up to 10 minutes
    st=$(cat $BUS/.voice_state 2>/dev/null); cap=$(cat $BUS/.voice_capturing 2>/dev/null)
    # a capturing flag stuck over a minute is a bug, not someone talking
    if [ "$cap" = 1 ]; then capped=$((capped + 1)); else capped=0; fi
    [ $capped -gt 120 ] && cap=0
    if [ "$st" = idle ] && [ "$cap" != 1 ]; then quiet=$((quiet + 1)); else quiet=0; fi
    [ $quiet -ge 10 ] && exec systemctl --user restart "${@:-jarvis-voice.service}"
    sleep 0.5
done
echo "never quiet for 5 s in 10 minutes; not restarting" >&2
exit 1
