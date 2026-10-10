import gzip, json, sys
ev=json.load(gzip.open(sys.argv[1]))['traceEvents']
gpu=sorted([e for e in ev if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset') and 'dur' in e], key=lambda e:e['ts'])
cpu=sorted([e for e in ev if e.get('cat')=='cpu_op' and 'dur' in e], key=lambda e:e['ts'])
t0=gpu[0]['ts']
big=[e['ts'] for e in gpu if e.get('cat')=='gpu_memcpy' and e['args'].get('bytes')==973209600]
print('w13 memcpys', len(big), 'first at %.0f ms'%((big[0]-t0)/1e3), 'last at %.0f'%((big[-1]-t0)/1e3), 'end %.0f'%((gpu[-1]['ts']+gpu[-1]['dur']-t0)/1e3))
# chunk boundaries: gaps between consecutive w13 > 300 ms
for a,b in zip(big,big[1:]):
    if (b-a)/1e3>150: print('  chunk boundary: w13 at %.0f -> next w13 at %.0f (%.0f ms)'%((a-t0)/1e3,(b-t0)/1e3,(b-a)/1e3))
# list gaps >= 15 ms with position, and cpu-op coverage
gaps=[]
for a,b in zip(gpu,gpu[1:]):
    g=b['ts']-(a['ts']+a['dur'])
    if g>=15000: gaps.append((a['ts']+a['dur'], b['ts'], a['name'][:40], b['name'][:40]))
def cov(s,e):
    best=None
    for c in cpu:
        if c['ts']<e and c['ts']+c['dur']>s:
            ov=min(e,c['ts']+c['dur'])-max(s,c['ts'])
            if best is None or ov>best[0]: best=(ov,c['name'][:40])
    return best
inloop=outloop=0
for s,e,an,bn in gaps:
    inside=any(abs(s-b)<120000 and s>=b for b in big)
    d=(e-s)/1e3
    if inside: inloop+=d
    else: outloop+=d
    if d>=25 or not inside:
        c=cov(s,e)
        print(f'  gap {d:6.1f} ms at {(s-t0)/1e3:7.0f} {"IN " if inside else "OUT"} after={an:40s} before={bn:40s} cpu={c}')
print(f'big-gap time inside layer loop {inloop:.0f} ms, outside {outloop:.0f} ms')
