"""
Release the SepsisGuard dev-server port.

The FastAPI server binds 127.0.0.1:8000 explicitly, so a server left running from an
earlier session makes the next start fail with:

    [Errno 10048] only one usage of each socket address ... is normally permitted

This script finds whatever is listening on the port and stops it. It is wired to the
Claude Code SessionEnd hook (.claude/settings.json) so the port is freed automatically
when a session finishes, and can also be run by hand:

    python scripts/free_port.py          # frees 8000
    python scripts/free_port.py 8001     # frees another port

Only processes owned by the current user are touched, and only those actually
listening on the given port.
"""

import subprocess
import sys


def listeners(port):
    """PIDs listening on `port`, via netstat (no third-party dependencies)."""
    try:
        out = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return set()

    pids = set()
    needle = f":{port}"
    for line in out.splitlines():
        parts = line.split()
        # Proto  Local Address  Foreign Address  State  PID
        if len(parts) >= 5 and parts[3].upper() == "LISTENING":
            local = parts[1]
            if local.rsplit(":", 1)[-1] == str(port) and needle in local:
                if parts[4].isdigit() and parts[4] != "0":
                    pids.add(int(parts[4]))
    return pids


def stop(pid):
    try:
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/F"],
            capture_output=True, text=True, timeout=30, check=False,
        )
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def main():
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Not a port number: {sys.argv[1]}")
            return 1

    pids = listeners(port)
    if not pids:
        print(f"Port {port} already free.")
        return 0

    for pid in sorted(pids):
        print(f"Freeing port {port}: stopping PID {pid}" if stop(pid)
              else f"Could not stop PID {pid} on port {port}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
