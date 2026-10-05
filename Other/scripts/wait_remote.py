"""Poll a rented host until a marker appears in a log, then exit.

Keeps long GPU runs from costing one check every ten minutes: start this in the
background and it prints the tail of the log once the run is finished.

  python scripts/wait_remote.py --host ssh4.vast.ai --port 38566 \
      --log /workspace/project/logs_q7b_queue.log --marker QUEUE_DONE
"""
from __future__ import annotations

import argparse
import subprocess
import time


def remote(host: str, port: int, command: str) -> tuple[int, str]:
    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=30", "-o", "StrictHostKeyChecking=no",
         "-p", str(port), f"root@{host}", command],
        capture_output=True, text=True)
    return result.returncode, result.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--marker", default="QUEUE_DONE")
    parser.add_argument("--interval", type=int, default=600)
    parser.add_argument("--max-hours", type=float, default=24.0)
    args = parser.parse_args()

    deadline = time.time() + args.max_hours * 3600
    failures = 0
    while time.time() < deadline:
        code, out = remote(args.host, args.port, f"cat {args.log} 2>/dev/null")
        if code != 0:
            failures += 1
            print(f"ssh unreachable ({failures})", flush=True)
        else:
            failures = 0
            if args.marker in out:
                print("MARKER FOUND\n" + out[-2000:], flush=True)
                return
        time.sleep(args.interval)
    print("TIMED OUT waiting for marker", flush=True)


if __name__ == "__main__":
    main()
