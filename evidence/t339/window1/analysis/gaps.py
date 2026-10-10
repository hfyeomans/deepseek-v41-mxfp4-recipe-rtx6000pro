import gzip, json, sys, collections, statistics
p=sys.argv[1]
ev=json.load(gzip.open(p))['traceEvents']
gpu=sorted([e for e in ev if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset') and 'dur' in e], key=lambda e:e['ts'])
cpu=sorted([e for e in ev if e.get('cat')=='cpu_op' and 'dur' in e], key=lambda e:e['ts'])
t0=gpu[0]['ts']; span=(gpu[-1]['ts']+gpu[-1]['dur']-t0)/1e3
busy=sum(e['dur'] for e in gpu)/1e3
print(f'span {span:.0f} ms, gpu busy {busy:.0f} ms, cpu_ops {len(cpu)}')
# per-layer period
big=sorted([e['ts'] for e in gpu if e.get('cat')=='gpu_memcpy' and e['args'].get('bytes')==973209600])
d=[(b-a)/1e3 for a,b in zip(big,big[1:]) if (b-a)/1e3<300]
print(f'layer period median {statistics.median(d):.1f} ms p10 {sorted(d)[len(d)//10]:.1f} p90 {sorted(d)[9*len(d)//10]:.1f} n={len(d)} sum {sum(d):.0f}')
gaps=[]
for a,b in zip(gpu,gpu[1:]):
    g=b['ts']-(a['ts']+a['dur'])
    if g>=1000: gaps.append((a['ts']+a['dur'], b['ts'], a['name'][:45], b['name'][:45]))
print('gaps>=1ms', len(gaps), 'total %.0f ms'%(sum((e-s) for s,e,_,_ in gaps)/1e3))
# what cpu ops are running during big gaps (>=15 ms): longest-overlap top-level ops
def top_ops(s,e):
    out=collections.Counter()
    for c in cpu:
        if c['ts']<e and c['ts']+c['dur']>s:
            ov=min(e,c['ts']+c['dur'])-max(s,c['ts'])
            out[c['name'][:70]]+=ov
    return out
agg=collections.Counter(); n=0; nocpu=0
for s,e,an,bn in gaps:
    if e-s<15000: continue
    n+=1
    t=top_ops(s,e)
    if not t: nocpu+=(e-s)
    for k,v in t.items(): agg[k]+=v
print(f'big gaps (>=15ms): {n}; time with no cpu_op at all: {nocpu/1e3:.0f} ms')
for k,v in agg.most_common(12): print(f'  {v/1e3:7.0f} ms  {k}')
# host duration of the fused moe op per call
fm=[c for c in cpu if 'fused_moe' in c['name'] or 'cutlass_fused' in c['name']]
if fm:
    ds=sorted(c['dur']/1e3 for c in fm)
    print(f'host-side fused-moe op: n={len(fm)} median {statistics.median(ds):.2f} ms p90 {ds[9*len(ds)//10]:.2f} max {ds[-1]:.1f} sum {sum(ds):.0f}; names {set(c["name"][:50] for c in fm)}')
# longest host ops overall
longest=sorted(cpu,key=lambda c:-c['dur'])[:8]
for c in longest: print(f'  host op {c["dur"]/1e3:7.1f} ms at {(c["ts"]-t0)/1e3:7.0f}  {c["name"][:80]}')
