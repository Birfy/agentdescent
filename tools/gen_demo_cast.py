"""Record `examples.run_demo` as an asciicast, for the README's animated SVG.

    python -m tools.gen_demo_cast                    # write the .cast
    python -m tools.gen_demo_cast --svg              # ...and convert it

The demo finishes in about half a second, which is the point of showing it: the
whole evolution loop runs with no API key and no network. A recording that short
is unwatchable on its own, so the cast holds the finished table on screen at the
end rather than cutting the moment the process exits.

What is real and what is not, because a demo that fakes its output is worthless:
the command runs for real through a pty, and every byte and every inter-chunk
delay below comes from that run. Only the keystrokes of the command line itself
are synthesised — nobody records themselves typing, and the alternative is a
recording that opens on a line that appeared from nowhere.

Needs `svg-term-cli` (npm) for `--svg`. Not a dependency of the package or the
suite; this is a maintainer tool, run when the demo's output changes.
"""

import argparse
import json
import os
import pty
import select
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAST = os.path.join(ROOT, "docs", "assets", "demo.cast")
SVG = os.path.join(ROOT, "docs", "assets", "demo.svg")
COMMAND = "python -m examples.run_demo"
COLS, ROWS = 100, 30
TYPE_DELAY = 0.055      # seconds per keystroke
HOLD = 2.5              # seconds to hold the final frame


def record(command: str) -> list:
    """Run `command` under a pty, returning [(elapsed, text)] as it arrives."""
    events, start = [], time.time()
    pid, fd = pty.fork()
    if pid == 0:                                   # child
        os.environ["TERM"] = "xterm-256color"
        os.environ["COLUMNS"], os.environ["LINES"] = str(COLS), str(ROWS)
        os.execvp("sh", ["sh", "-c", f"cd {ROOT} && {command}"])
    while True:
        try:
            ready, _, _ = select.select([fd], [], [], 30)
        except (OSError, ValueError):
            break
        if not ready:
            break
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            break
        if not chunk:
            break
        events.append((time.time() - start, chunk.decode("utf-8", "replace")))
    os.close(fd)
    os.waitpid(pid, 0)
    return events


def build(events: list) -> str:
    """An asciicast v2 document: a JSON header, then one JSON array per write."""
    out = [json.dumps({"version": 2, "width": COLS, "height": ROWS,
                       "timestamp": int(time.time()), "idle_time_limit": 2,
                       "env": {"SHELL": "/bin/sh", "TERM": "xterm-256color"}})]

    t = 0.0
    def emit(text):
        out.append(json.dumps([round(t, 6), "o", text]))

    emit("$ ")
    for ch in COMMAND:                             # synthesised keystrokes
        t += TYPE_DELAY
        emit(ch)
    t += 0.35
    emit("\r\n")

    offset = t
    for elapsed, text in events:                   # real output, real timing
        t = offset + elapsed
        emit(text)

    t += HOLD
    emit("")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--svg", action="store_true", help="also convert with svg-term")
    args = ap.parse_args(argv)

    events = record(COMMAND)
    if not events:
        raise SystemExit("the demo produced no output; nothing to record")
    os.makedirs(os.path.dirname(CAST), exist_ok=True)
    with open(CAST, "w", encoding="utf-8") as fh:
        fh.write(build(events))
    span = events[-1][0]
    print(f"wrote {os.path.relpath(CAST, ROOT)} "
          f"({len(events)} writes, {span:.2f}s of real run time)")

    if args.svg:
        exe = shutil.which("svg-term") or os.path.join(
            ROOT, "node_modules", ".bin", "svg-term")
        if not shutil.which(exe) and not os.path.exists(exe):
            raise SystemExit("svg-term not found: npm install -g svg-term-cli")
        subprocess.run([exe, "--in", CAST, "--out", SVG, "--window",
                        "--width", str(COLS), "--height", str(ROWS),
                        "--term", "iterm2", "--profile", "Seti"], check=True)
        print(f"wrote {os.path.relpath(SVG, ROOT)} "
              f"({os.path.getsize(SVG) // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
