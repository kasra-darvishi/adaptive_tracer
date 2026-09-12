#!/usr/bin/env python3
import argparse
import subprocess
import re
import lttngust.loghandler
import logging
import select
import time
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--quiet", action="store_true", help="Suppress printing spans to stdout")
args = parser.parse_args()

logger = logging.getLogger("otel.spans")
logger.handlers.clear()
lttng_handler = lttngust.loghandler._Handler()
logger.addHandler(lttng_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

# Matches OTel Java agent LoggingSpanExporter output
span_pattern = re.compile(
    r"\[otel\.javaagent.*?\]\s+INFO.*?LoggingSpanExporter - '(.+?)' : ([a-f0-9]+) ([a-f0-9]+) (\w+)"
)

# Docker logs --timestamps prefix, e.g. 2026-03-23T12:34:56.123456789Z ...
docker_ts_pattern = re.compile(
    r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)\s+(.*)$"
)

services = ["carts", "orders", "shipping", "queue-master"]
containers = {f"docker-compose_{s}_1": s for s in services}

procs = {}
stream_to_meta = {}

for container, service in containers.items():
    p = subprocess.Popen(
        ["docker", "logs", "-f", "--tail=0", "--timestamps", container],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    procs[container] = p
    stream_to_meta[p.stdout] = {"container": container, "service": service}

if not args.quiet:
    print("Relaying OTel spans to LTTng... (Ctrl+C to stop)")

try:
    while True:
        readable, _, _ = select.select(list(stream_to_meta.keys()), [], [], 1.0)
        for stream in readable:
            raw = stream.readline()
            if not raw:
                continue

            line = raw.decode("utf-8", errors="replace").strip()
            meta = stream_to_meta[stream]
            service = meta["service"]
            container = meta["container"]

            docker_ts = ""
            m_ts = docker_ts_pattern.match(line)
            if m_ts:
                docker_ts, line = m_ts.groups()

            m = span_pattern.search(line)
            if not m:
                continue

            op, trace_id, span_id, kind = m.groups()
            relay_ts_ns = time.time_ns()

            # Keep op quoted in case it contains spaces
            safe_op = op.replace("\\", "\\\\").replace('"', '\\"')

            msg = (
                f'service={service} '
                f'container={container} '
                f'phase=export '
                f'docker_ts={docker_ts or "na"} '
                f'relay_ts_ns={relay_ts_ns} '
                f'trace_id={trace_id} '
                f'span_id={span_id} '
                f'kind={kind} '
                f'op="{safe_op}"'
            )

            logger.info(msg)

            if not args.quiet:
                print(f"[LTTng] {msg}")

except KeyboardInterrupt:
    if not args.quiet:
        print("Stopped.")
finally:
    for p in procs.values():
        try:
            p.terminate()
        except Exception:
            pass
    for p in procs.values():
        try:
            p.wait(timeout=2)
        except Exception:
            pass