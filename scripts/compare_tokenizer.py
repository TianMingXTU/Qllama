"""Compare the Rust tokenizer with recorded llama.cpp IDs and check text recovery."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-boundaries", action="store_true")
    args = parser.parse_args()
    evidence = json.loads(args.oracle.read_text(encoding="utf-8"))
    model = Path(evidence["model"]["path"])
    with model.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != evidence["model"]["sha256"]:
        raise ValueError("Model SHA256 does not match oracle evidence")
    cases = evidence["cases"]
    env = {key: value for key, value in os.environ.items() if not key.startswith("LLAMA_ARG_")}
    if args.probe_boundaries:
        probes = {
            "crlf": "first\r\nsecond\r\n",
            "bare_cr": "first\rsecond",
            "nul": "before\u0000after",
            "metaspace": "a▁b",
            "metaspace_space_twin": "a b",
            "non_bmp": "𐐀𠀀𝄞",
            "combining_unicode": "é é ñ",
            "space_around_special": "Hello </s> world",
        }
        for name, prompt in probes.items():
            command = [evidence["oracle"]["executable"]] + evidence["command"]["common_args"]
            oracle = subprocess.run(command, input=prompt.encode("utf-8"), stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, check=True, env=env)
            cases.append({"name": name, "prompt": prompt, "args": [], "ids": json.loads(oracle.stdout)})
    run = subprocess.run(
        [str(args.ours.resolve()), str(model)],
        input=json.dumps([case["prompt"] for case in cases], ensure_ascii=False).encode("utf-8"),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    ours = json.loads(run.stdout)
    results = []
    unsupported = []
    for case, actual in zip(cases, ours, strict=True):
        if "--no-parse-special" in case["args"]:
            unsupported.append(case["name"])
            continue
        expected = case["ids"]
        transport_recheck = None
        # Windows 文本模式 stdin 会把 CRLF 转为 LF；用 argv 复查原始输入。
        if "\r\n" in case["prompt"]:
            common = evidence["command"]["common_args"]
            command = [evidence["oracle"]["executable"]] + [x for x in common if x != "--stdin"]
            command += case["args"] + ["-p", case["prompt"]]
            oracle = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, env=env)
            expected = json.loads(oracle.stdout)
            transport_recheck = {"argv": command, "stdin_ids": case["ids"], "exact_prompt_ids": expected}
        ids = actual["plain_ids"] if "--no-bos" in case["args"] else actual["ids"]
        match = ids == expected
        round_trip = actual["decoded"] == case["prompt"]
        result = {
            "name": case["name"], "prompt": case["prompt"], "args": case["args"],
            "ours": ids, "llamacpp": expected, "ids_match": match,
            "decoded": actual["decoded"], "round_trip": round_trip,
            "raw_bytes": actual["raw_bytes"], "transport_recheck": transport_recheck,
        }
        results.append(result)
        print(f"{case['name']}: ids={len(ids)}/{len(expected)} match={match} round_trip={round_trip}")
        if not match:
            print(f"  ours={ids}\n  cpp ={expected}")
    report = {
        "model_sha256": digest, "oracle_version": evidence["oracle"]["version"],
        "cases": results, "unsupported_parse_special_false_cases": unsupported,
        "matched": sum(row["ids_match"] for row in results),
        "round_tripped": sum(row["round_trip"] for row in results),
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"IDS_MATCH={report['matched']}/{len(results)} ROUND_TRIP={report['round_tripped']}/{len(results)}")
    if not all(row["ids_match"] for row in results):
        raise SystemExit(1)
    # 只有实测与空格版本完全同 id 的 U+2581 输入，才能归为 SPM 不可逆归一化。
    for row in results:
        if row["round_trip"]:
            continue
        twins = [other for other in results
                 if other["prompt"] == row["prompt"].replace("▁", " ")
                 and other["args"] == row["args"]]
        if "▁" not in row["prompt"] or not any(other["ours"] == row["ours"] for other in twins):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
