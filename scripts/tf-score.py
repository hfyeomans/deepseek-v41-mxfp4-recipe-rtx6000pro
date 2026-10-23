#!/usr/bin/env python3
"""E35 substitute quality gate: paired teacher-forced next-token logprob.

Decoder SWA bounded replay refuses prompt logprobs (only the last 128 rows of an
extend are computed past layer 20), so the T3.28 corpus NLL cannot run on it.
This scores p(x_k | x_<k) for sampled k via /generate with input_ids=x_<k,
max_new_tokens=1 and token_ids_logprob=[x_k] (reads output_token_ids_logprobs).
Three request shapes, matching the two production shapes plus the worst case:

  fresh   /flush_cache before each request -> full prefill of k tokens
  ext     positions k0+1..k0+N in order; the radix hit is page-aligned (256), so
          the extend length cycles 1..256 (chat-turn shape; 1 is the worst case)

Token ids come from a corpus JSON dir written by scripts/nll-score.py --corpus
(per-text token_ids). Output: one JSON with per-shape per-position logprobs.
--compare pools (B - A) per shape over common positions and prints mean / se.
"""
import argparse
import json
import math
import pathlib
import random
import sys
import time
import urllib.request

sys.path.insert(0, "scripts")
from evidence_manifest import stamp_or_exit  # noqa: E402


def post(url, path, body, timeout=600):
    req = urllib.request.Request(f"{url}{path}", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def flush(url):
    req = urllib.request.Request(f"{url}/flush_cache", data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        resp.read()


def tf_logprob(url, ids, k):
    body = {"input_ids": ids[:k], "sampling_params": {"max_new_tokens": 1, "temperature": 0},
            "return_logprob": True, "token_ids_logprob": [ids[k]], "top_logprobs_num": 0}
    r = post(url, "/generate", body)
    meta = r["meta_info"]
    ent = meta["output_token_ids_logprobs"][0][0]
    return {"k": k, "true_id": ids[k], "logprob": ent[0], "greedy_id": r.get("output_ids", [None])[0],
            "cached": meta.get("cached_tokens"), "prompt_tokens": meta.get("prompt_tokens")}


def load_corpus(d):
    texts = {}
    for p in sorted(pathlib.Path(d).glob("*.json")):
        if p.name == "pooled.json":
            continue
        j = json.loads(p.read_text())
        ids = j.get("token_ids")
        if ids:
            texts[p.stem] = ids
    return texts


def run(args):
    url = args.url.rstrip("/")
    texts = load_corpus(args.corpus_json_dir)
    rng = random.Random(args.seed)
    out = {"shapes": {}, "corpus_json_dir": args.corpus_json_dir, "seed": args.seed}
    t0 = time.time()
    # fresh: fixed positions + random ones, each after /flush_cache
    fresh = {}
    for name, ids in texts.items():
        n = len(ids)
        ks = [k for k in (256, 512, 1024, 2048, 3072, 4096) if k < n - 1]
        ks += sorted(rng.sample(range(128, min(n - 1, 4096)), args.fresh_random))
        rows = []
        for k in ks:
            flush(url)
            rows.append(tf_logprob(url, ids, k))
        fresh[name] = rows
        print(f"fresh {name}: {len(rows)} positions, mean lp {sum(r['logprob'] for r in rows)/len(rows):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    out["shapes"]["fresh"] = fresh
    # ext1 / ext32: prefill k0 once, then cache-hit extends of 1 / 32 tokens
    for shape, step, count in (("ext", 1, args.ext_count),):
        res = {}
        for name, ids in texts.items():
            n = len(ids)
            k0 = args.k0
            if k0 + step * count >= n - 1:
                count_t = (n - 2 - k0) // step
            else:
                count_t = count
            flush(url)
            tf_logprob(url, ids, k0)  # warm the radix cache to k0
            rows = []
            for j in range(1, count_t + 1):
                rows.append(tf_logprob(url, ids, k0 + step * j))
            res[name] = rows
            ext = [r["prompt_tokens"] - (r["cached"] or 0) for r in rows if r["prompt_tokens"] is not None]
            print(f"{shape} {name}: {len(rows)} positions, mean lp {sum(r['logprob'] for r in rows)/len(rows):.4f}, "
                  f"extend lens min/max {min(ext) if ext else '?'}/{max(ext) if ext else '?'} ({time.time()-t0:.0f}s)",
                  flush=True)
        out["shapes"][shape] = res
    out["elapsed_s"] = round(time.time() - t0, 1)
    out["label"] = args.label
    outp = pathlib.Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    stamp_or_exit(out, outp, "e35-tf-score.py")
    outp.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.output}")
    if args.compare:
        compare(args.output, args.compare)


def compare(cand_path, base_path):
    cand = json.loads(pathlib.Path(cand_path).read_text())
    base = json.loads(pathlib.Path(base_path).read_text())
    print(f"\n== paired delta logp (candidate - baseline), nats/token; + = worse for the candidate's NLL? no: "
          f"delta logp < 0 means the candidate assigns LOWER probability (worse)")
    summary = {}
    for shape in cand["shapes"]:
        if shape not in base["shapes"]:
            continue
        d = []
        same = 0
        for name, rows in cand["shapes"][shape].items():
            b = {r["k"]: r for r in base["shapes"][shape].get(name, [])}
            for r in rows:
                if r["k"] in b and b[r["k"]]["true_id"] == r["true_id"]:
                    d.append(r["logprob"] - b[r["k"]]["logprob"])
                    same += int(r["greedy_id"] == b[r["k"]]["greedy_id"])
        n = len(d)
        if n == 0:
            continue
        if shape == "ext":
            bins = {"1-8": [], "9-32": [], "33-128": [], "129-256": []}
            for name, rows in cand["shapes"][shape].items():
                b = {r["k"]: r for r in base["shapes"][shape].get(name, [])}
                for r in rows:
                    if r["k"] in b and r["prompt_tokens"] is not None:
                        e = r["prompt_tokens"] - (r["cached"] or 0)
                        key = "1-8" if e <= 8 else "9-32" if e <= 32 else "33-128" if e <= 128 else "129-256"
                        bins[key].append(r["logprob"] - b[r["k"]]["logprob"])
            for key, v in bins.items():
                if v:
                    m = sum(v) / len(v); sd = math.sqrt(sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1))
                    print(f"  ext len {key:8s} n={len(v):4d} dNLL={-m:+.4f} se={sd/math.sqrt(len(v)):.4f} worst={min(v):+.3f}")
            summary["ext_bins"] = {k: {"n": len(v), "delta_nll": -sum(v)/len(v) if v else None} for k, v in bins.items()}
        mean = sum(d) / n
        var = sum((x - mean) ** 2 for x in d) / max(n - 1, 1)
        se = math.sqrt(var / n)
        worst = min(d)
        big = sum(1 for x in d if abs(x) > 0.5)
        summary[shape] = {"n": n, "mean_delta_logp": mean, "se": se, "delta_nll": -mean,
                          "worst_delta_logp": worst, "abs_gt_0.5": big, "greedy_same": same}
        print(f"{shape:6s} n={n:5d} dNLL={-mean:+.4f} se={se:.4f} 95%CI=[{-mean-1.96*se:+.4f},{-mean+1.96*se:+.4f}] "
              f"worst dlogp={worst:+.3f} |d|>0.5: {big} greedy-same {same}/{n}")
    cand["compare"] = {"baseline": base_path, "summary": summary}
    stamp_or_exit(cand, pathlib.Path(cand_path), "tf-score.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--corpus-json-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--compare")
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--seed", type=int, default=35)
    ap.add_argument("--fresh-random", type=int, default=14)
    ap.add_argument("--k0", type=int, default=1024)
    ap.add_argument("--ext-count", type=int, default=300)
    a = ap.parse_args()
    if a.compare_only:
        compare(a.output, a.compare)
    else:
        run(a)
