#!/usr/bin/env python3
"""HumanEval pass@1, greedy, on the served hybrid (E32).

Two phases, both from this script:

  generate  164 problems through /v1/chat/completions, temperature 0, one sample each, thinking
            off, max_tokens 4096, stop on the model's own stop. Per request: TTFT, decode tok/s
            (first-to-last token), completion tokens, finish reason. Writes samples.jsonl plus a
            program file per problem (model code + the dataset's check()).
  score     runs every program inside a throwaway container with no network (--network none,
            default image dsv41-kt:recipe, mounted read-only), one subprocess per problem with a
            wall timeout; a problem passes when check() exits 0.

Both phases write report.json, stamped with the serve's evidence manifest like the other served
scripts. The six problems the mlx-stream V4.1-Flash recipe fails (32, 83, 132, 140, 145, 163) are
reported separately so the two numbers can be compared on the other 158.

Dataset: data/HumanEval.jsonl.gz from github.com/openai/human-eval (MIT), expected at
~/.cache/dsv41-recipe/humaneval/HumanEval.jsonl.gz; --dataset overrides.

    python3 scripts/bench-humaneval.py --url http://127.0.0.1:8000 --model dsv41-flash-hybrid \
        --output-dir evidence/e32-humaneval
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import hashlib
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evidence_manifest import stamp_or_exit  # noqa: E402
from openai_client import chat_stream, get_json  # noqa: E402

DEFAULT_DATASET = Path.home() / ".cache" / "dsv41-recipe" / "humaneval" / "HumanEval.jsonl.gz"
MLX_STREAM_FAILS = (32, 83, 132, 140, 145, 163)
INSTRUCTION = (
    "Complete the following Python function. Reply with the complete, runnable function (its imports, "
    "signature and body) in one ```python code block and nothing else.\n\n```python\n{prompt}\n```"
)
FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.S)

# Executed inside the scoring container. Runs every *.py under /work/programs with a per-problem
# timeout and writes /out/results.json. stdlib only; no network is reachable in that container.
RUNNER = r'''
import json, os, subprocess, sys
progs = sorted(os.listdir("/work/programs"))
timeout = float(sys.argv[1])
out = {}
for name in progs:
    path = os.path.join("/work/programs", name)
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
        out[name[:-3]] = {"passed": p.returncode == 0, "returncode": p.returncode,
                          "stderr_tail": p.stderr[-600:]}
    except subprocess.TimeoutExpired:
        out[name[:-3]] = {"passed": False, "returncode": None, "stderr_tail": "timeout"}
with open("/out/results.json", "w") as f:
    json.dump(out, f, indent=1)
'''


def load_problems(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:  # type: ignore[operator]
        return [json.loads(line) for line in f if line.strip()]


def task_index(task_id: str) -> int:
    return int(task_id.split("/")[-1])


def extract_program(problem: dict, text: str) -> tuple[str, str]:
    """Last fenced code block; if it does not define the entry point, treat it as a completion of
    the dataset prompt (prompt + code), the classic HumanEval mode. Returns (program, mode)."""
    blocks = FENCE.findall(text)
    code = blocks[-1] if blocks else text
    code = code.rstrip() + "\n"
    if re.search(rf"^\s*def\s+{re.escape(problem['entry_point'])}\s*\(", code, re.M):
        return code, "full-function"
    return problem["prompt"] + code, "completion"


def generate(args, model: str, problems: list[dict], out: Path) -> dict:
    programs = out / "programs"
    programs.mkdir(parents=True, exist_ok=True)

    def one(problem: dict) -> dict:
        messages = [{"role": "user", "content": INSTRUCTION.format(prompt=problem["prompt"].rstrip())}]
        r = chat_stream(args.url, model, messages, max_tokens=args.max_tokens, temperature=0.0, seed=args.seed,
                        thinking=args.thinking)
        program, mode = extract_program(problem, r.text)
        return {"task_id": problem["task_id"], "text": r.text, "program": program, "mode": mode,
                "reasoning_chars": len(r.reasoning),
                "ttft_s": r.ttft_s, "total_s": r.total_s, "decode_tps": r.decode_tps(),
                "completion_tokens": r.completion_tokens, "prompt_tokens": r.prompt_tokens,
                "finish_reason": r.finish_reason}

    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        samples = list(ex.map(one, problems))
    wall = time.perf_counter() - t0
    with (out / "samples.jsonl").open("w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    by_id = {p["task_id"]: p for p in problems}
    for s in samples:
        p = by_id[s["task_id"]]
        name = f"{task_index(s['task_id']):03d}.py"
        (programs / name).write_text(s["program"] + "\n\n" + p["test"] + f"\n\ncheck({p['entry_point']})\n")
    tps = [s["decode_tps"] for s in samples if s["decode_tps"]]
    ttft = [s["ttft_s"] for s in samples]
    toks = [s["completion_tokens"] or 0 for s in samples]
    gen = {"wall_s": wall, "n": len(samples), "concurrency": args.concurrency, "max_tokens": args.max_tokens,
           "seed": args.seed, "temperature": 0.0, "thinking": args.thinking,
           "median_reasoning_chars": statistics.median([s["reasoning_chars"] for s in samples]),
           "median_decode_tps": statistics.median(tps) if tps else None,
           "min_decode_tps": min(tps) if tps else None, "max_decode_tps": max(tps) if tps else None,
           "median_ttft_s": statistics.median(ttft), "max_ttft_s": max(ttft),
           "median_completion_tokens": statistics.median(toks), "total_completion_tokens": sum(toks),
           "finish_reasons": {k: sum(1 for s in samples if s["finish_reason"] == k)
                              for k in sorted({str(s["finish_reason"]) for s in samples})},
           "modes": {k: sum(1 for s in samples if s["mode"] == k) for k in ("full-function", "completion")}}
    print(f"generated {len(samples)} in {wall:.0f} s: decode median {gen['median_decode_tps']:.2f} tok/s "
          f"(min {gen['min_decode_tps']:.2f}), TTFT median {gen['median_ttft_s']:.3f} s, "
          f"finish {gen['finish_reasons']}, modes {gen['modes']}")
    return gen


def score(args, out: Path) -> dict:
    programs = out / "programs"
    if not programs.is_dir():
        raise SystemExit(f"no programs under {out}; run the generate phase first")
    (out / "runner.py").write_text(RUNNER)
    cmd = ["docker", "run", "--rm", "--network", "none", "--entrypoint", "python3",
           "-v", f"{out.resolve()}:/work:ro", "-v", f"{out.resolve()}:/out",
           args.image, "/work/runner.py", str(args.timeout)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"scoring container failed rc={p.returncode}\n{p.stderr[-2000:]}")
    results = json.loads((out / "results.json").read_text())
    n = len(results)
    passed = sorted(int(k) for k, v in results.items() if v["passed"])
    failed = sorted(int(k) for k, v in results.items() if not v["passed"])
    mlx = set(MLX_STREAM_FAILS)
    common = [i for i in range(n) if i not in mlx]
    sc = {"image": args.image, "timeout_s": args.timeout, "n": n, "passed": len(passed), "failed": failed,
          "pass_at_1": len(passed) / n if n else None,
          "mlx_stream_fails": {"ids": list(MLX_STREAM_FAILS),
                               "ours_passed": sorted(i for i in MLX_STREAM_FAILS if i in passed)},
          "excluding_mlx_stream_fails": {"n": len(common), "passed": sum(1 for i in common if i in passed)},
          "failed_detail": {str(i): results[str(i).zfill(3)]["stderr_tail"].strip().splitlines()[-1:]
                            for i in failed}}
    print(f"pass@1 {len(passed)}/{n} = {sc['pass_at_1']:.4f}; failed {failed}; "
          f"on mlx-stream's 6 failures we pass {sc['mlx_stream_fails']['ours_passed']}; "
          f"excluding those six: {sc['excluding_mlx_stream_fails']['passed']}/{len(common)}")
    return sc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None)
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=1, help="1 keeps per-request decode/TTFT single-stream")
    ap.add_argument("--limit", type=int, default=None, help="first N problems (smoke test)")
    ap.add_argument("--thinking", action="store_true",
                    help="thinking on (long-reasoning decode traffic; the answer is still the last fenced block of the "
                         "content, the reasoning is kept per sample as a length only)")
    ap.add_argument("--phase", choices=("all", "generate", "score"), default="all")
    ap.add_argument("--image", default="dsv41-kt:recipe", help="image for the no-network scoring container")
    ap.add_argument("--timeout", type=float, default=20.0, help="per-problem wall timeout in the scorer")
    args = ap.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    if args.phase in ("all", "generate"):
        if not args.dataset.exists():
            raise SystemExit(f"dataset not found: {args.dataset} (data/HumanEval.jsonl.gz from "
                             f"github.com/openai/human-eval)")
        problems = load_problems(args.dataset)[: args.limit]
        model = args.model or get_json(f"{args.url}/v1/models")["data"][0]["id"]
        report = {"url": args.url, "model": model, "dataset": str(args.dataset).replace(str(Path.home()), "~"),
                  "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(), "n_problems": len(problems),
                  "method": f"chat completion, greedy (temperature 0), one sample, thinking {'on' if args.thinking else 'off'}, last fenced block "
                            "(+ dataset prompt when the block lacks the entry point) run against check() in a "
                            "no-network container",
                  "generate": generate(args, model, problems, out)}
        report["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        stamp_or_exit(report, report_path, "bench-humaneval.py")
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    if args.phase in ("all", "score"):
        if "generate" not in report:
            raise SystemExit(f"{report_path} has no generate phase; run --phase generate first")
        report["score"] = score(args, out)
        report["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        stamp_or_exit(report, report_path, "bench-humaneval.py")
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
