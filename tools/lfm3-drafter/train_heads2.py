import sys, struct, math, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
GGUF="/home/majed/models/lfm3/LFM3-1.7B-Q8_0.gguf"
K=int(sys.argv[1]); EPOCHS=float(sys.argv[2]); PREFIX=sys.argv[3]; OUT=sys.argv[4]
H=2048; dev="cuda"
def read_q8_embedding():
    f=open(GGUF,"rb"); assert f.read(4)==b"GGUF"
    rd=lambda fmt: struct.unpack(fmt,f.read(struct.calcsize(fmt)))[0]
    ver=rd("<I"); nt=rd("<Q"); nk=rd("<Q")
    def rstr(): n=rd("<Q"); return f.read(n).decode("utf8","replace")
    def rval(t):
        T={0:"<B",1:"<b",2:"<H",3:"<h",4:"<I",5:"<i",6:"<f",7:"<?",10:"<Q",11:"<q",12:"<d"}
        if t in T: return rd(T[t])
        if t==8: return rstr()
        if t==9:
            et=rd("<I"); n=rd("<Q"); return [rval(et) for _ in range(n)]
    align=32
    for _ in range(nk):
        k=rstr(); t=rd("<I"); v=rval(t)
        if k=="general.alignment": align=v
    info={}
    for _ in range(nt):
        n=rstr(); d=rd("<I"); sh=[rd("<Q") for _ in range(d)]; ty=rd("<I"); off=rd("<Q"); info[n]=(sh,ty,off)
    pos=f.tell(); base=(pos+align-1)//align*align
    sh,ty,off=info["token_embd.weight"]
    f.seek(base+off); raw=np.frombuffer(f.read(128000*H//32*34),dtype=np.uint8).reshape(-1,34)
    scale=raw[:,:2].copy().view(np.float16).astype(np.float32)
    q=raw[:,2:].view(np.int8).astype(np.float32)
    return (q*scale).reshape(128000,H)
E=torch.from_numpy(read_q8_embedding()).to(dev).to(torch.bfloat16)
toks=np.fromfile(PREFIX+".tokens",dtype=np.uint32).astype(np.int64)
hid=np.memmap(PREFIX+".hidden",dtype=np.float16,mode="r").reshape(-1,H)
starts=np.fromfile(PREFIX+".docs",dtype=np.uint64).astype(np.int64)
ends=np.append(starts[1:],len(toks))
# im_start id = token after BOS in every doc
im_start=np.bincount(toks[starts+1]).argmax()
print("im_start id",im_start,flush=True)
pos_train=[];pos_held=[]
nd=len(starts); held_docs=max(40,nd//25)
for d,(s,e) in enumerate(zip(starts,ends)):
    seg=toks[s:e]
    ims=np.nonzero(seg==im_start)[0]
    if len(ims)==0: continue
    plen=ims[-1]+5
    lo=s+plen-1; hi=e-2-K   # t such that t+2+K-1 < e
    if hi<=lo: continue
    (pos_held if d>=nd-held_docs else pos_train).append(np.arange(lo,hi+1))
pos_train=np.concatenate(pos_train); pos_held=np.concatenate(pos_held)
print("train positions",len(pos_train),"held positions",len(pos_held),flush=True)
class Head(nn.Module):
    def __init__(s): super().__init__(); s.w=nn.Linear(H,H,bias=False); nn.init.zeros_(s.w.weight)
    def forward(s,h): return h+F.silu(s.w(h))
heads=nn.ModuleList([Head() for _ in range(K)]).to(dev)
tok_t=torch.from_numpy(toks).to(dev)
B=2048; steps=int(EPOCHS*len(pos_train)/B)
opt=torch.optim.AdamW(heads.parameters(),lr=1e-3,weight_decay=0.0)
sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=1e-3,total_steps=steps,pct_start=0.05)
rng=np.random.default_rng(0); t0=time.time()
for step in range(steps):
    idx=np.sort(rng.choice(pos_train,B,replace=False))
    h=torch.from_numpy(hid[idx]).to(dev).float(); loss=0
    for j in range(K):
        logits=heads[j](h).to(torch.bfloat16)@E.T
        y=tok_t[torch.from_numpy(idx+2+j).to(dev)]
        loss=loss+F.cross_entropy(logits.float(),y)
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    if step%100==0: print(f"step {step}/{steps} loss {loss.item()/K:.3f} {time.time()-t0:.0f}s",flush=True)
sd=heads.state_dict()
with open(OUT,"wb") as f:
    f.write(struct.pack("<II",K,H))
    for j in range(K): f.write(sd[f"{j}.w.weight"].to(torch.float16).cpu().contiguous().numpy().tobytes())
with torch.no_grad():
    res=np.zeros((K,len(pos_held)),dtype=bool)
    for lo in range(0,len(pos_held),4096):
        p=pos_held[lo:lo+4096]; h=torch.from_numpy(hid[p]).to(dev).float()
        for j in range(K):
            pred=(heads[j](h).to(torch.bfloat16)@E.T).argmax(-1).cpu().numpy()
            res[j,lo:lo+len(p)]=pred==toks[p+2+j]
run=np.ones(len(pos_held),dtype=bool); expect=0.0; accs=[]
for j in range(K):
    run=run&res[j]; accs.append(res[j].mean()); expect+=run.mean()
print("held head top-1:",[f"{a:.3f}" for a in accs],"expected extra accepted/pass (chain):",f"{expect:.2f}",flush=True)
