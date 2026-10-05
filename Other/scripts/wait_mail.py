"""Poll the agent inbox every N seconds; print and exit when a new message arrives."""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mail  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since-uid", type=int, required=True)
    parser.add_argument("--interval", type=int, default=180)
    args = parser.parse_args()
    while True:
        buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(buffer):
                mail.check(args.since_uid, show_all=False)
        except Exception as exc:  # network hiccups should not end the watcher
            print(f"check failed: {exc}", flush=True)
        else:
            if "=== UID" in buffer.getvalue():
                print(buffer.getvalue(), flush=True)
                return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
