#!/usr/bin/env bash
# T3.41: one developer serve at launcher defaults + SGLANG_DSV41_FP8_GEMV=1 (overlay d00e8e30).
# Decode benches (the GEMV only dispatches at M <= 4), NLL quality + identity, same-prompt C4.
set -u
D=evidence/t341/serve-gemv
U=http://127.0.0.1:8000
M=dsv41-flash-hybrid
health() { docker logs dsv41-kt 2>&1 | rg -i -e " nan" -e "probability tensor" -e Xid -e Traceback -e KTContract -e "Sealed.*Error" -e OutOfMemory | rg -v "any later grow raises" | wc -l; }
echo "== benches $(date +%T)"
docker logs dsv41-kt 2>&1 | rg "FP8 GEMV table" > $D/gemv-table-log.txt
nvidia-smi --query-gpu=memory.used,memory.total --format=csv > $D/vram-after-capture.txt
python3 scripts/bench-decode.py --url $U --model $M --concurrency 1 --fixture-offset 6 --max-tokens 512 --runs 6 --warmup 1 --distinct-prompts --seed 4242 --output $D/c1.json > $D/c1.txt 2>&1; tail -1 $D/c1.txt
python3 scripts/bench-decode.py --url $U --model $M --concurrency 4 --fixture-offset 6 --max-tokens 512 --runs 3 --warmup 1 --distinct-prompts --seed 4242 --output $D/c4.json > $D/c4.txt 2>&1; tail -1 $D/c4.txt
python3 scripts/bench-decode.py --url $U --model $M --concurrency 2 --fixture-offset 6 --max-tokens 512 --runs 3 --warmup 1 --distinct-prompts --seed 4242 --output $D/c2.json > $D/c2.txt 2>&1; tail -1 $D/c2.txt
python3 scripts/bench-decode.py --url $U --model $M --concurrency 3 --fixture-offset 6 --max-tokens 512 --runs 3 --warmup 1 --distinct-prompts --seed 4242 --output $D/c3.json > $D/c3.txt 2>&1; tail -1 $D/c3.txt
echo "== nll $(date +%T)"
python3 scripts/nll-score.py --url $U --corpus scripts/fixtures/nll-corpus --output-dir $D/corpus --compare-dir evidence/t328/default-219-8192/corpus > $D/corpus.txt 2>&1; tail -2 $D/corpus.txt
python3 scripts/nll-score.py --url $U --corpus scripts/fixtures/nll-corpus --output-dir $D/corpus-vs-anchor --compare-dir evidence/t315c/default-flip/corpus > $D/corpus-vs-anchor.txt 2>&1; tail -2 $D/corpus-vs-anchor.txt
python3 scripts/verify-batch-consistency.py --url $U --model $M --concurrency 4 --same-prompt --label t341-gemv --output $D/consistency.json > $D/consistency.txt 2>&1; tail -2 $D/consistency.txt
echo "health=$(health)"
cp "$(ls -t ~/.cache/dsv41-recipe/manifests/*-dsv41-kt.json | head -1)" $D/manifest.json
echo "== done $(date +%T)"
