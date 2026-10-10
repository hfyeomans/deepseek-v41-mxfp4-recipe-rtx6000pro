import gzip, json, sys
p,s0,e0=sys.argv[1],float(sys.argv[2]),float(sys.argv[3])
ev=json.load(gzip.open(p))['traceEvents']
gpu=sorted([e for e in ev if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset') and 'dur' in e], key=lambda e:e['ts'])
t0=gpu[0]['ts']
cpu=sorted([e for e in ev if e.get('cat') in ('cpu_op','user_annotation','python_function') and 'dur' in e and s0*1e3<=e['ts']-t0<=e0*1e3], key=lambda e:e['ts'])
print('cats in window:', {e.get('cat') for e in cpu})
# top-level (non-nested) ops: skip ops fully inside previous op
stack=[]
for c in cpu:
    while stack and stack[-1]['ts']+stack[-1]['dur']<=c['ts']: stack.pop()
    depth=len(stack); stack.append(c)
    if c['dur']>=3000 and depth<=1:
        print(f"{(c['ts']-t0)/1e3:8.0f} +{c['dur']/1e3:7.1f} ms d{depth} tid{c.get('tid')} {c['name'][:90]} {str(c.get('args',{}).get('Input Dims',''))[:60]}")
print('--- gpu events in window >= 2ms or memcpy')
for g in gpu:
    if s0*1e3<=g['ts']-t0<=e0*1e3 and (g['dur']>=2000 or 'Memcpy' in g['name']):
        print(f"{(g['ts']-t0)/1e3:8.0f} +{g['dur']/1e3:6.1f} {g['name'][:50]} bytes={g['args'].get('bytes')}")
