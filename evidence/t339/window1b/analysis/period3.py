import gzip, json, sys, collections
p=sys.argv[1]; ev=json.load(gzip.open(p))['traceEvents']
gpu=sorted([e for e in ev if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset') and 'dur' in e], key=lambda e:e['ts'])
w13=sorted(e['ts'] for e in gpu if e.get('cat')=='gpu_memcpy' and e['args'].get('bytes')==973209600)
k=int(sys.argv[2]); a=w13[k]
lo,hi=float(sys.argv[3]),float(sys.argv[4])
host=sorted([e for e in ev if e.get('cat') not in ('kernel','gpu_memcpy','gpu_memset','gpu_user_annotation','ac2g') and 'dur' in e and a+lo*1000<=e['ts']<=a+hi*1000], key=lambda e:e['ts'])
for c in host:
    print(f'{(c["ts"]-a)/1e3:8.2f} {c["dur"]/1e3:7.2f} tid{c["tid"]} {str(c.get("cat")):16s} {c["name"][:90]}')
