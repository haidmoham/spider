"""Export the explicitly published checkpoint for the portfolio synchronizer.

This reads preserved evidence; it does not run a policy or choose a promotion.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_feed():
    selection = json.loads((ROOT / "public/selection.json").read_text(encoding="utf-8"))
    policy = selection["policy"]
    if Path(policy).name != policy:
        raise ValueError("Policy must name an artifact directory")
    directory = ROOT / "artifacts" / policy
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if "accepted" not in manifest["status"] or manifest["capability"]["status"] != "complete":
        raise ValueError("Only an accepted, completed checkpoint can be published")
    weights = manifest["checkpoint"]
    if hashlib.sha256((directory / weights).read_bytes()).hexdigest() != manifest["sha256"][weights]:
        raise ValueError("Preserved checkpoint hash does not match")
    media = {}
    for filename_key, url_key, hash_key in (("video", "url", "sha256"), ("poster", "poster_url", "poster_sha256")):
        name = selection[filename_key]
        if Path(name).name != name:
            raise ValueError("Media must be a file directly under public/")
        data = (ROOT / "public" / name).read_bytes()
        media[url_key] = f"https://raw.githubusercontent.com/haidmoham/spider/master/public/{name}"
        media[hash_key] = hashlib.sha256(data).hexdigest()
    copy_values = {
        "policy": policy, "mean_speed_m_s": manifest["mean_speed_m_s"],
        "falls": manifest["falls"], "evaluations": manifest["evaluation_count"],
    }
    return {
        "schema_version": 1,
        "checkpoint": {
            "id": selection["id"], "label": selection["label"], "policy": policy,
            "source_commit": selection["source_commit"],
            "evidence_url": f"https://github.com/haidmoham/spider/blob/{selection['source_commit']}/{selection['evidence_path']}",
        },
        "metrics": {
            "mean_speed_m_s": manifest["mean_speed_m_s"],
            "sampled_mean_speed_m_s": manifest["sampled_mean_speed_m_s"],
            "falls": manifest["falls"], "evaluations": manifest["evaluation_count"],
        },
        "copy": {key: value.format_map(copy_values) for key, value in selection["copy"].items()}, "scope": selection["scope"],
        "limits": selection["limits"], "media": media,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    output = ROOT / "public/checkpoint.json"
    encoded = json.dumps(build_feed(), ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if output.read_text(encoding="utf-8") != encoded:
            raise SystemExit("Public checkpoint feed is stale; export it before publishing")
        print("Public checkpoint feed matches accepted evidence and media hashes")
    else:
        output.write_text(encoded, encoding="utf-8")
        print(output)
