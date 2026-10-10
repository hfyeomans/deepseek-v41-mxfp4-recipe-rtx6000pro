import gzip, json, sys
p=sys.argv[1]; ev=json.load(gzip.open(p))['traceEvents']
gpu=sorted([e for e in ev if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset') and 'dur' in e], key=lambda e:e['ts'])
w13=sorted(e['ts'] for e in gpu if e.get('cat')=='gpu_memcpy' and e['args'].get('bytes')==973209600)
k=int(sys.argv[2]); a=w13[k]; b=w13[k+1]
seg=[e for e in gpu if a-100<=e['ts']<b]
# merge busy intervals
busy=[];cs=ce=None
for e in seg:
    s,t=e['ts'],e['ts']+e['dur']
    if ce is None or s>ce+50:
        if ce is not None: busy.append((cs,ce,names))
        cs,ce,names=s,t,[e['name'][:38]]
    else:
        ce=max(ce,t); names.append(e['name'][:38])
busy.append((cs,ce,names))
for s,t,n in busy:
    print(f'{(s-a)/1e3:7.1f} -> {(t-a)/1e3:7.1f} busy {(t-s)/1e3:6.1f} ms  n={len(n)} first={n[0]} last={n[-1]}')
