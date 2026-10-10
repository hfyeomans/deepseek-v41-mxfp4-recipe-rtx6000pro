import gzip, json, sys, collections
p=sys.argv[1]; ev=json.load(gzip.open(p))['traceEvents']
gpu=sorted([e for e in ev if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset') and 'dur' in e], key=lambda e:e['ts'])
w13=sorted(e['ts'] for e in gpu if e.get('cat')=='gpu_memcpy' and e['args'].get('bytes')==973209600)
k=int(sys.argv[2]) if len(sys.argv)>2 else len(w13)//2
a,b=w13[k],w13[k+1]
print(f'period {k}: {(b-a)/1e3:.1f} ms')
host=sorted([e for e in ev if e.get('cat') in ('cuda_runtime','cpu_op','cuda_driver') and 'dur' in e and e['ts']<b+1000 and e['ts']+e['dur']>a-40000], key=lambda e:e['ts'])
tids=collections.Counter(e['tid'] for e in host); print('tids',tids.most_common(4))
tid=tids.most_common(1)[0][0]
host=[e for e in host if e['tid']==tid]
top=[];end=-1
for c in host:
    if c['ts']>=end: top.append(c); end=c['ts']+c['dur']
prev=None
for c in top:
    off=(c['ts']-a)/1e3
    if prev is not None and c['ts']-prev>1500:
        print(f'   -- host idle {(c["ts"]-prev)/1e3:6.1f} ms (ends at {off:7.1f})')
    if c['dur']>300:
        print(f'{off:8.1f} {c["dur"]/1e3:7.1f} ms {c["cat"]:12s} {c["name"][:60]} {str(c.get("args",{}).get("bytes",""))}')
    prev=c['ts']+c['dur']
# GPU events in the period, >2 ms
print('GPU >2ms:')
for e in gpu:
    if a-1 <= e['ts'] < b and e['dur']>2000:
        print(f'{(e["ts"]-a)/1e3:8.1f} {e["dur"]/1e3:7.1f} ms {e["cat"]:10s} {e["name"][:60]} {e["args"].get("bytes","")}')
