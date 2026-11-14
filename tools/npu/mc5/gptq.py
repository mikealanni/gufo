import sys,os,glob,time,numpy as np,torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(int(os.environ.get('GPTQ_THREADS','16')))
d='/home/majed/jev-model/MiniCPM5-1B-hf'
out=os.environ.get('GPTQ_OUT','gptq')
tok=AutoTokenizer.from_pretrained(d)
text=''
for f in sorted(glob.glob('/home/majed/gufo-wp10/docs/*.md'))+sorted(glob.glob('/home/majed/gufo-wp10/src/models/*/*.md')):
    text+=open(f,errors='ignore').read()+'\n'
ids=[0]+tok(text,add_special_tokens=False)['input_ids']
CAL_START=8192; WIN=int(os.environ.get('GPTQ_WIN','512')); NW=int(os.environ.get('GPTQ_NW','128'))
m=AutoModelForCausalLM.from_pretrained(d,dtype=torch.float32).eval()
H={}
cnt={}
def hook(name):
    def f(mod,inp):
        x=inp[0].reshape(-1,inp[0].shape[-1]).double()
        H[name]=H.get(name,0)+x.T@x
        cnt[name]=cnt.get(name,0)+x.shape[0]
    return f
for l,layer in enumerate(m.model.layers):
    layer.self_attn.q_proj.register_forward_pre_hook(hook(f'{l}.qkv'))
    layer.self_attn.o_proj.register_forward_pre_hook(hook(f'{l}.o'))
    layer.mlp.gate_proj.register_forward_pre_hook(hook(f'{l}.gu'))
    layer.mlp.down_proj.register_forward_pre_hook(hook(f'{l}.down'))
t0=time.time()
for w in range(NW):
    s=CAL_START+w*WIN
    x=torch.tensor([ids[s:s+WIN]])
    with torch.no_grad(): m(x)
print('hessians',time.time()-t0,flush=True)
def bf16r(t): return (t.view(torch.int32)+0x7FFF+((t.view(torch.int32)>>16)&1)>>16<<16).view(torch.float32) if False else t.to(torch.bfloat16).float()
def gptq(W,Hm,bs=128,damp=0.01):
    W=W.clone().double(); N,K=W.shape
    Hm=Hm.clone()/1.0
    dead=torch.diag(Hm)==0
    Hm[dead,dead]=1; W[:,dead]=0
    Hm+=damp*torch.mean(torch.diag(Hm))*torch.eye(K,dtype=torch.float64)
    L=torch.linalg.cholesky(Hm); Hinv=torch.cholesky_inverse(L); Hinv=torch.linalg.cholesky(Hinv,upper=True)
    Q=torch.zeros(N,K,dtype=torch.int8); S=torch.zeros(N,K//32,dtype=torch.float32)
    for i1 in range(0,K,bs):
        i2=min(i1+bs,K); W1=W[:,i1:i2].clone(); E1=torch.zeros_like(W1); H1=Hinv[i1:i2,i1:i2]
        for i in range(i2-i1):
            if (i1+i)%32==0:
                amax=W1[:,i:i+32].abs().amax(1).float()
                sc=bf16r(amax/15.0); sc[sc==0]=1.0
                S[:,(i1+i)//32]=sc
                scd=sc.double()
            w=W1[:,i]; dd=H1[i,i]
            q=torch.clamp(torch.round(w/scd),-15,15)
            Q[:,i1+i]=q.to(torch.int8)
            e=(w-q*scd)/dd
            W1[:,i:]-=e[:,None]*H1[i,i:][None,:]
            E1[:,i]=e
        W[:,i2:]-=E1@Hinv[i1:i2,i2:]
    return Q,S
def save(name,Q,S):
    np.save(f'{out}/{name}.q.npy',Q.numpy())
    np.save(f'{out}/{name}.s.npy',(S.contiguous().view(torch.int32)>>16).to(torch.int16).numpy().astype(np.uint16))
def rtn_err(W,Q,S):
    N,K=W.shape
    D=(Q.float().reshape(N,K//32,32)*S[:,:,None]).reshape(N,K)
    return ((W-D)**2).sum().item()
os.makedirs(out,exist_ok=True)
t0=time.time()
for l,layer in enumerate(m.model.layers):
    jobs=[('q',layer.self_attn.q_proj.weight,f'{l}.qkv'),('k',layer.self_attn.k_proj.weight,f'{l}.qkv'),('v',layer.self_attn.v_proj.weight,f'{l}.qkv'),
          ('o',layer.self_attn.o_proj.weight,f'{l}.o'),('gate',layer.mlp.gate_proj.weight,f'{l}.gu'),('up',layer.mlp.up_proj.weight,f'{l}.gu'),
          ('down',layer.mlp.down_proj.weight,f'{l}.down')]
    for nm,Wt,hk in jobs:
        W=Wt.detach().float()
        Q,S=gptq(W,H[hk]/cnt[hk])
        save(f'{l}.{nm}',Q,S)
    print('layer',l,time.time()-t0,flush=True)
