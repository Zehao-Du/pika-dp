#!/usr/bin/env bash
set -u

PATTERN='python3 (eval_real_pika.py|scripts_real/eval_pika.py|umi/real_world/pika_env.py)'

show_processes() {
    pgrep -af "$PATTERN" || true
}

kill_with_signal() {
    local signal="$1"
    local pids
    pids="$(pgrep -f "$PATTERN" || true)"
    if [[ -z "$pids" ]]; then
        return 0
    fi

    echo "Sending SIG${signal} to:"
    ps -o pid,ppid,stat,etime,cmd -p $(echo "$pids" | tr '\n' ' ') || true
    kill "-${signal}" $pids || true
}

echo "Current eval processes:"
show_processes

kill_with_signal INT
sleep 1

if pgrep -f "$PATTERN" >/dev/null; then
    kill_with_signal TERM
    sleep 1
fi

if pgrep -f "$PATTERN" >/dev/null; then
    kill_with_signal KILL
    sleep 0.5
fi

echo
echo "Remaining eval processes:"
show_processes

echo
echo "GPU processes:"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true
else
    echo "nvidia-smi not found"
fi
