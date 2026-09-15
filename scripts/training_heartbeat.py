"""Report every ten minutes or fifty updates; never starts training."""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import time


def snapshot(roots):
    latest = 200
    evaluation = None
    modified = None
    for root in roots:
        try:
            path = root / "updates.csv"
            with path.open() as stream:
                updates = [int(row["update"]) for row in csv.DictReader(stream)]
            if updates and max(updates) >= latest:
                latest, modified = max(updates), path.stat().st_mtime
        except (FileNotFoundError, KeyError, ValueError):
            pass  # The trainer may currently be replacing its CSV.
        for path in sorted(root.glob("comparison-n*/n*/acceptance.json")):
            try:
                result = json.loads(path.read_text())
                records = result["records"]
                mean = next(row for row in records if row["mode"] == "mean")
                if evaluation is None or mean["updates"] > evaluation["update"]:
                    evaluation = dict(update=mean["updates"], mean_m_s=mean["speed_m_s"],
                                      sampled_m_s=result["sampled_mean_speed_m_s"],
                                      falls=sum(row["terminated"] for row in records),
                                      trials=len(records))
            except (OSError, ValueError, KeyError, StopIteration):
                pass
    return dict(time_utc=datetime.now(timezone.utc).isoformat(), last_saved_update=latest,
                seconds_since_update=None if modified is None else round(time.time()-modified),
                latest_evaluation=evaluation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--interval", type=float, default=600)
    parser.add_argument("--until", type=int, default=500)
    parser.add_argument("--every-updates", type=int, default=50)
    args = parser.parse_args()
    if args.interval <= 0 or args.every_updates <= 0:
        parser.error("interval and every-updates must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 7200
    next_report = 0.0
    last_bucket = None
    while time.monotonic() < deadline:
        receipt = snapshot(args.roots)
        evaluation = receipt["latest_evaluation"]
        finished = evaluation is not None and evaluation["update"] >= args.until
        bucket = receipt["last_saved_update"] // args.every_updates
        if finished or bucket != last_bucket or time.monotonic() >= next_report:
            receipt["reason"] = ("evaluation_complete" if finished else
                                 "update_milestone" if last_bucket is not None and bucket != last_bucket
                                 else "heartbeat")
            with args.output.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(receipt) + "\n")
            print(json.dumps(receipt), flush=True)
            last_bucket = bucket
            next_report = time.monotonic() + args.interval
        if finished:
            break
        time.sleep(5)


if __name__ == "__main__":
    main()
