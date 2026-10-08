#!/bin/bash
# G1 exit soak (migration-roadmap ledger, Stage 1 exit gate): one C1..C4 soak block, E27 Boot 12 / E34a shape
# (8 bench blocks x 12 runs x 512 tokens, ~49K decoded tokens) on the final Stage 1 overlay commit, launcher
# defaults (219 / 165, D=4, spec off, chunk 8192, 64K x 4). Plus the per-step gate: same-prompt consistency at
# concurrency 1..4, NLL bit-identical to nll-t316c.json, prefill 16K as a cold / warm pair (T3.19f), 4 x 62K
# stress and the 64K needles. Every JSON is stamped with the serve's manifest (first stamped evidence).
set -u
cd ~/dev/src/deepseek-v41-fp4-recipe-rtx6000pro
D=evidence/t319/g1-exit/soak
U=http://127.0.0.1:8000; M=dsv41-flash-hybrid
health() { docker logs dsv41-kt 2>&1 | rg -i -e nan -e "probability tensor" -e Xid -e Traceback -e KTContract -e Sealed | rg -v "Decode batch|Prefill batch|NAN_PROBE|any later grow raises" | wc -l; }
echo "== start $(date +%T) edac=$(cat /sys/devices/system/edac/mc/mc*/{ce,ue}_count 2>/dev/null | tr '\n' ' ')"
nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,temperature.gpu,memory.used --format=csv -lms 2000 > $D/gpu-during-soak.csv 2>/dev/null &
GPU_PID=$!
python3 scripts/hsmp-clocks.py --watch 5 --n 1200 > $D/hsmp-during-soak.txt 2>&1 &
HSMP_PID=$!
for c in 1 2 3 4; do
  echo "== consistency c$c $(date +%T)"
  python3 scripts/verify-batch-consistency.py --url $U --model $M --concurrency $c --same-prompt --label g1-c$c --output $D/consistency-c$c.json 2>&1 | tail -1
done
echo health=$(health)
for blk in c4-off0 c3-off2 c2-off4 c4-off8 c1-off6 c4-off12 c3-off9 c2-off0; do
  c=${blk:1:1}; off=${blk#*off}
  echo "== $blk $(date +%T)"
  python3 scripts/bench-decode.py --url $U --model $M --max-tokens 512 --runs 12 --warmup 0 --distinct-prompts --seed 4242 --concurrency $c --fixture-offset $off --output $D/$blk.json 2>&1 | tail -2
  echo health=$(health)
done
echo "== nll $(date +%T)"
python3 scripts/nll-score.py --url $U --compare evidence/t316-review-intake/nll-t316c.json --label g1-exit --output $D/nll.json 2>&1 | tail -1
echo "== prefill-16k cold+warm $(date +%T)"
python3 scripts/bench-prefill.py --url $U --model $M --tokens 16384 --runs 2 --warm-pass --output $D/prefill-16k.json 2>&1 | tail -6
echo "== stress 4x62K $(date +%T)"
python3 scripts/stress-kv-full.py --url $U --model $M --concurrency 4 --prompt-tokens 63488 --max-tokens 256 --output $D/stress-kv.json 2>&1 | tail -4
echo "== needles 64k $(date +%T)"
python3 scripts/verify.py --url $U --model $M --context 65536 --output $D/verify-64k.json 2>&1 | tail -2
kill $GPU_PID $HSMP_PID 2>/dev/null
echo "== done $(date +%T) health=$(health) edac=$(cat /sys/devices/system/edac/mc/mc*/{ce,ue}_count 2>/dev/null | tr '\n' ' ')"
