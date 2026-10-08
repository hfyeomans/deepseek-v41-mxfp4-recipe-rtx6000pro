#!/bin/bash
# E34a 64K x 4-slot served soak, E27 Boot 12 shape (8 bench blocks x 12 runs x 512 tokens) + the 64K-specific gates.
set -u
cd ~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro
D=evidence/e34-indexer-memory/a/soak-64k-4slot
U=http://127.0.0.1:8000; M=dsv41-flash-hybrid
health() { docker logs dsv41-kt 2>&1 | rg -i -e nan -e "probability tensor" -e Xid -e Traceback -e KTContract -e Sealed | rg -v "Decode batch|Prefill batch|NAN_PROBE|any later grow raises" | wc -l; }
echo "== verify-c4 $(date +%T)"
python3 scripts/verify-batch-consistency.py --url $U --model $M --concurrency 4 --same-prompt --label soak-c4 > $D/verify-c4.json 2>$D/verify-c4.err; echo verify_exit=$? health=$(health)
for blk in c4-off0 c3-off2 c2-off4 c4-off8 c1-off6 c4-off12 c3-off9 c2-off0; do
  c=${blk:1:1}; off=${blk#*off}
  echo "== $blk $(date +%T)"
  python3 scripts/bench-decode.py --url $U --model $M --max-tokens 512 --runs 12 --warmup 0 --distinct-prompts --seed 4242 --concurrency $c --fixture-offset $off --output $D/$blk.json 2>&1 | tail -2
  echo health=$(health)
done
echo "== nll $(date +%T)"
python3 scripts/nll-score.py --url $U --compare evidence/t316-review-intake/nll-t316c.json --label soak-64k4 --output $D/nll.json 2>&1 | tail -1
echo "== prefill-16k $(date +%T)"
python3 scripts/bench-prefill.py --url $U --model $M --tokens 16384 --runs 1 --output $D/prefill-16k.json 2>&1 | tail -2
echo "== stress 4x62K $(date +%T)"
python3 scripts/stress-kv-full.py --url $U --model $M --concurrency 4 --prompt-tokens 63488 --max-tokens 256 --output $D/stress-kv.json 2>&1 | tail -4
echo "== needles 64k $(date +%T)"
python3 scripts/verify.py --url $U --model $M --context 65536 --output $D/verify-64k.json 2>&1 | tail -2
echo "== done $(date +%T) health=$(health) edac=$(cat /sys/devices/system/edac/mc/mc*/{ce,ue}_count | tr '\n' ' ')"
