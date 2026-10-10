#!/usr/bin/env python3
"""Teacher-forced mean NLL of a fixed text: the quality metric for approximations.

Greedy-token comparison (greedy-logprobs.py) turned out to be noise on this
hybrid: at D=0, two bs=1 runs of the same prompt already pick different tokens
at near-ties, so "diverges at token N" says nothing about deferral or kernel
quality (E21 result 7). Teacher-forced NLL does not have that problem: the
tokens are fixed, and averaging −log p over a few thousand of them turns tie
flips into a few 1e-4 nats of noise.

Uses SGLang's native /generate with return_logprob + logprob_start_len=0; the
response carries meta_info.input_token_logprobs = [[logprob, token_id, None],
...] (first entry has logprob None). Scores the frozen corpus
scripts/fixtures/nll-corpus.txt (≈3.5K tokens) unless --text-file is given.

  # baseline on config A (run twice to see the noise floor)
  python3 scripts/nll-score.py --output evidence/x/nll-d0.json
  # on config B, compare
  python3 scripts/nll-score.py --output evidence/x/nll-d2.json --compare evidence/x/nll-d0.json

Reports mean NLL (nats/token), per-1K-token segments, and vs the baseline:
ΔNLL, max per-token |Δlogp|, and how many tokens changed by >0.5 nats.

Corpus mode (T3.28, shape-aware gate). One fixture is ~4.8K tokens, and with
block-scaled FP8/FP4 activations any change in kernel schedule (prefill chunk,
expert placement, GEMM tile) reorders accumulations and adds ~0.1 nats/token of
zero-mean jitter, so a single text has se ≈ 0.0035 — bigger than the +0.002
gate. --corpus scores every *.txt in a directory (scripts/fixtures/nll-corpus/,
≈60K tokens across in-house docs and scripts), writes one JSON per text plus
pooled.json, and when --compare-dir names a baseline directory it pools the
per-token Δlogp over all texts (paired, common token-id prefix only):

  python3 scripts/nll-score.py --corpus scripts/fixtures/nll-corpus \
      --output-dir evidence/t328/baseline-226-4096
  python3 scripts/nll-score.py --corpus scripts/fixtures/nll-corpus \
      --output-dir evidence/t328/cand-219-8192 --compare-dir evidence/t328/baseline-226-4096

Gate: same kernel shapes -> every token bit-identical (tokens_changed == 0);
shape-changing -> pooled delta_mean_nll <= +0.002 nats/token, reported with se.
"""
import argparse
import json
import math
import pathlib
import sys
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from evidence_manifest import stamp_or_exit  # noqa: E402
from openai_client import auth_headers  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_TEXT = HERE / "fixtures" / "nll-corpus.txt"
REPO = HERE.parent


def portable(path) -> str:
    """Record paths relative to the repo root so evidence JSON carries no home directory."""
    p = pathlib.Path(path).resolve()
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(path)


def score(url: str, text: str, timeout: float) -> dict:
    body = {
        "text": text,
        "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        "return_logprob": True,
        "logprob_start_len": 0,
        "top_logprobs_num": 0,
    }
    req = urllib.request.Request(f"{url.rstrip('/')}/generate", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", **auth_headers()})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        r = json.load(resp)
    meta = r["meta_info"]
    entries = meta["input_token_logprobs"]
    lps = [e[0] for e in entries[1:]]  # position 0 has no prediction
    ids = [e[1] for e in entries]
    if any(lp is None for lp in lps):
        raise SystemExit("server returned None logprobs after position 0; is return_logprob supported?")
    return {"token_ids": ids, "logprobs": lps, "prompt_tokens": meta.get("prompt_tokens"),
            "cached_tokens": meta.get("cached_tokens"), "e2e_latency_s": meta.get("e2e_latency")}


def common_prefix_len(a: list, b: list) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def paired_stats(new_lps: list[float], base_lps: list[float], n: int | None = None) -> dict:
    """Paired per-token comparison over the first n tokens (default: shorter list).

    delta_mean_nll = -mean(new_logp - base_logp): positive = worse than baseline.
    z = mean / (sd / sqrt(n)) treats tokens as independent draws, which for the
    ~0.1-nat accumulation-order jitter they are to a good approximation.
    """
    if n is None:
        n = min(len(new_lps), len(base_lps))
    d = [new_lps[i] - base_lps[i] for i in range(n)]
    if n == 0:
        return {"n": 0, "delta_mean_nll": 0.0, "se": 0.0, "z_paired": 0.0, "max_abs_delta_logp": 0.0,
                "mean_abs_delta_logp": 0.0, "sd_delta_logp": 0.0, "tokens_changed": 0,
                "tokens_worse_gt_0p5": 0, "tokens_better_gt_0p5": 0}
    mean_d = sum(d) / n
    sd = math.sqrt(sum((x - mean_d) ** 2 for x in d) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n)
    return {"n": n, "delta_mean_nll": -mean_d, "se": se, "z_paired": mean_d / se if se else 0.0,
            "max_abs_delta_logp": max(abs(x) for x in d), "mean_abs_delta_logp": sum(abs(x) for x in d) / n,
            "sd_delta_logp": sd, "tokens_changed": sum(1 for x in d if x != 0.0),
            "tokens_worse_gt_0p5": sum(1 for x in d if x < -0.5),
            "tokens_better_gt_0p5": sum(1 for x in d if x > 0.5)}


def pooled_stats(pairs: list[tuple[list[float], list[float], int]]) -> dict:
    """Pool per-token deltas across texts: pairs of (new_lps, base_lps, n_common)."""
    new_all: list[float] = []
    base_all: list[float] = []
    for new_lps, base_lps, n in pairs:
        new_all.extend(new_lps[:n])
        base_all.extend(base_lps[:n])
    st = paired_stats(new_all, base_all)
    # Cluster-robust se with one cluster per text: per-token deltas are not iid across a text
    # (one prefill batch shape per chunk), so the iid se understates the uncertainty of the mean.
    n_tot = st["n"]
    if n_tot:
        mean_d = -st["delta_mean_nll"]
        acc = 0.0
        for new_lps, base_lps, n in pairs:
            acc += (sum(new_lps[i] - base_lps[i] for i in range(n)) - n * mean_d) ** 2
        st["se_cluster"] = math.sqrt(acc) / n_tot
        st["z_cluster"] = mean_d / st["se_cluster"] if st["se_cluster"] else 0.0
        st["texts"] = len(pairs)
    return st


def compare_to(res: dict, summ: dict, base: dict, base_name: str) -> dict:
    n = common_prefix_len(base["token_ids"], res["token_ids"])
    if n != len(res["token_ids"]) or n != len(base["token_ids"]):
        print(f"WARNING: tokenisation differs from baseline after token {n}; comparing only the common prefix")
    # logprobs start at position 1, so n token ids give n-1 paired logprobs
    st = paired_stats(res["logprobs"], base["logprobs"], max(n - 1, 0))
    return {"baseline": base_name, "baseline_mean_nll": base["mean_nll"],
            "delta_full_text_nll": summ["mean_nll"] - base["mean_nll"], **st}


def print_compare(c: dict) -> None:
    print(f"vs {pathlib.Path(c['baseline']).name}: ΔNLL {c['delta_mean_nll']:+.5f} ± {c['se']:.5f} nats/tok "
          f"(z {c['z_paired']:+.2f}, n {c['n']})  max|Δlogp| {c['max_abs_delta_logp']:.3f}  "
          f"mean|Δlogp| {c['mean_abs_delta_logp']:.4f} sd {c['sd_delta_logp']:.3f}  changed {c['tokens_changed']}  "
          f"tokens >0.5 nats worse/better: {c['tokens_worse_gt_0p5']}/{c['tokens_better_gt_0p5']}")


def score_text(url: str, text_file: pathlib.Path, max_chars: int, seg: int, timeout: float, label: str) -> tuple[dict, dict, dict]:
    text = text_file.read_text()
    if max_chars:
        text = text[:max_chars]
    res = score(url, text, timeout)
    summ = summarize(res["logprobs"], seg)
    out = {"label": label, "text_file": str(text_file.name), "chars": len(text), **res, **summ}
    print(f"{text_file.name}: tokens scored {summ['tokens_scored']}  mean NLL {summ['mean_nll']:.5f} nats/tok  "
          f"ppl {summ['ppl']:.3f}  cached_tokens {res['cached_tokens']}  latency {res['e2e_latency_s']:.1f} s")
    return res, summ, out


def summarize(lps: list[float], seg: int) -> dict:
    n = len(lps)
    mean = -sum(lps) / n
    segments = []
    for i in range(0, n, seg):
        chunk = lps[i:i + seg]
        segments.append({"start": i, "n": len(chunk), "nll": -sum(chunk) / len(chunk)})
    return {"tokens_scored": n, "mean_nll": mean, "ppl": math.exp(mean), "segments": segments}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--text-file", type=pathlib.Path, default=DEFAULT_TEXT)
    ap.add_argument("--max-chars", type=int, default=0, help="truncate the text (0 = whole file)")
    ap.add_argument("--segment", type=int, default=1024, help="tokens per reported segment")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--output", type=pathlib.Path)
    ap.add_argument("--compare", type=pathlib.Path, help="baseline JSON from an earlier run")
    ap.add_argument("--label", default="", help="free-text tag saved in the JSON (e.g. 'D=2')")
    ap.add_argument("--corpus", type=pathlib.Path, help="directory of *.txt to score (T3.28 pooled gate)")
    ap.add_argument("--output-dir", type=pathlib.Path, help="corpus mode: write <stem>.json per text + pooled.json")
    ap.add_argument("--compare-dir", type=pathlib.Path, help="corpus mode: baseline directory from an earlier --corpus run")
    args = ap.parse_args()
    if args.corpus and not args.output_dir:
        ap.error("--corpus requires --output-dir")
    if (args.output_dir or args.compare_dir) and not args.corpus:
        ap.error("--output-dir/--compare-dir only apply with --corpus")

    if args.corpus:
        return run_corpus(args)

    res, summ, out = score_text(args.url, args.text_file, args.max_chars, args.segment, args.timeout, args.label)
    for sg in summ["segments"]:
        print(f"  seg @{sg['start']:>5} n={sg['n']:>4}  nll {sg['nll']:.5f}")

    if args.compare:
        base = json.loads(args.compare.read_text())
        out["compare"] = compare_to(res, summ, base, portable(args.compare))
        print_compare(out["compare"])

    if args.output:
        stamp_or_exit(out, args.output, "nll-score.py")
        args.output.write_text(json.dumps(out, indent=1))
        print(f"wrote {args.output}")
    return 0


def run_corpus(args) -> int:
    texts = sorted(p for p in args.corpus.iterdir() if p.suffix == ".txt")
    if not texts:
        raise SystemExit(f"no *.txt in {args.corpus}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_text = []
    pairs = []
    for tf in texts:
        res, summ, out = score_text(args.url, tf, 0, args.segment, args.timeout, args.label)
        row = {"text": tf.name, "tokens_scored": summ["tokens_scored"], "mean_nll": summ["mean_nll"]}
        if args.compare_dir:
            bp = args.compare_dir / f"{tf.stem}.json"
            if not bp.exists():
                raise SystemExit(f"baseline {bp} missing; score the baseline on the same corpus first")
            base = json.loads(bp.read_text())
            out["compare"] = compare_to(res, summ, base, portable(bp))
            print_compare(out["compare"])
            row["compare"] = out["compare"]
            pairs.append((res["logprobs"], base["logprobs"], out["compare"]["n"]))
        stamp_or_exit(out, args.output_dir / f"{tf.stem}.json", "nll-score.py")
        (args.output_dir / f"{tf.stem}.json").write_text(json.dumps(out, indent=1))
        per_text.append(row)

    pooled = {"label": args.label, "corpus": portable(args.corpus), "texts": per_text,
              "tokens_scored_total": sum(r["tokens_scored"] for r in per_text),
              "mean_nll_token_weighted": sum(r["mean_nll"] * r["tokens_scored"] for r in per_text)
              / sum(r["tokens_scored"] for r in per_text)}
    if args.compare_dir:
        pooled["compare"] = {"baseline_dir": portable(args.compare_dir), **pooled_stats(pairs)}
        c = pooled["compare"]
        print(f"POOLED over {len(per_text)} texts: ΔNLL {c['delta_mean_nll']:+.5f} ± {c['se']:.5f} nats/tok "
              f"(z {c['z_paired']:+.2f}, n {c['n']}; cluster-robust se {c['se_cluster']:.5f}, z {c['z_cluster']:+.2f})  "
              f"changed {c['tokens_changed']}  "
              f"worse/better >0.5: {c['tokens_worse_gt_0p5']}/{c['tokens_better_gt_0p5']}")
    else:
        print(f"corpus: {len(per_text)} texts, {pooled['tokens_scored_total']} tokens, "
              f"token-weighted mean NLL {pooled['mean_nll_token_weighted']:.5f}")
    stamp_or_exit(pooled, args.output_dir / "pooled.json", "nll-score.py")
    (args.output_dir / "pooled.json").write_text(json.dumps(pooled, indent=1))
    print(f"wrote {args.output_dir}/*.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
