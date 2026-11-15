import sys,os,time,numpy as np,torch
from safetensors import safe_open
torch.set_num_threads(int(os.environ.get('GPTQ_THREADS','24')))
out='gptq'
H=torch.load('hess.pt')
fs=[safe_open(f'/home/majed/pplx-model/base/model-0000{i}-of-00002.safetensors','pt') for i in (1,2)]
def get(name):
    for f in fs:
        if name in f.keys(): return f.get_tensor(name)
def bf16r(t): return t.to(torch.bfloat16).float()
def gptq(W,Hm,bs=128,damp=0.01):
    W=W.clone().double(); N,K=W.shape
    Hm=Hm.double().clone()
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
                S[:,(i1+i)//32]=sc; scd=sc.double()
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
os.makedirs(out,exist_ok=True)
t0=time.time()
for l in range(32):
    pre=f'language_model.layers.{l}.'
    lin=(l%4)!=3
    jobs=[]
    if lin:
        a=pre+'linear_attn.'
        jobs+=[('qkv',a+'in_proj_qkv.weight','in'),('z',a+'in_proj_z.weight','in'),('out',a+'out_proj.weight','out')]
    else:
        a=pre+'self_attn.'
        jobs+=[('q',a+'q_proj.weight','in'),('k',a+'k_proj.weight','in'),('v',a+'v_proj.weight','in'),('o',a+'o_proj.weight','out')]
    jobs+=[('gate',pre+'mlp.gate_proj.weight','gu'),('up',pre+'mlp.up_proj.weight','gu'),('down',pre+'mlp.down_proj.weight','dn')]
    for nm,wn,hk in jobs:
        W=get(wn).float()
        Hm=H[f'{l}.{hk}']/H[f'{l}.{hk}#n']
        Q,S=gptq(W,Hm)
        save(f'{l}.{nm}',Q,S)
    print('layer',l,time.time()-t0,flush=True)
