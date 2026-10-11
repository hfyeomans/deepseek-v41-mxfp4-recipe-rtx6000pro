import torch,sys
d=torch.load(sys.argv[1],map_location="cpu",weights_only=False)
t=d['trigger']; meta=t['meta']
i=[k for k in range(t['hits']) if int(meta[k,1])==36][0]
row=t['logits'][i][:14884]
v=row.to(torch.float16).view(torch.int16).to(torch.int32)&0xFFFF
bins=(v>>4)  # top 12 bits
kth=row.sort(descending=True).values[511]
kb=int((kth.to(torch.float16).view(torch.int16).to(torch.int32)&0xFFFF)>>4)
print("kth",float(kth),"bin",kb,"count in bin",int((bins==kb).sum()),"count above (fp16 bin >) : ",int(((row.to(torch.float16).float())>kth.to(torch.float16).float()).sum()))
# fp32-exact count greater than kth and equal
print("count > kth:",int((row>kth).sum()),"== kth:",int((row==kth).sum()))
# values sorted around boundary
s=row.sort(descending=True).values
print([float(x) for x in s[505:520]])
