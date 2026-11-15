import sys,json,glob,random,numpy as np,torch,time
sys.path.insert(0,'/home/majed/pplx-model/base/source/src')
from pplx_decider.model import Decider
random.seed(7)
d=Decider.from_pretrained('/home/majed/pplx-model/base',device='cpu',locate=False,cpu_threads=16)
text=''
for f in sorted(glob.glob('/home/majed/gufo-wp10/docs/*.md'))+sorted(glob.glob('/home/majed/gufo-wp10/src/models/*/*.md'))+sorted(glob.glob('/home/majed/gufo-wp10/tools/npu/*/*.cpp'))[:6]:
    text+=open(f,errors='ignore').read()+'\n'
words=text.split()
noul=["Is the issue urgent?","Does this require a code change?","Is the statement correct?","Is this about performance?","Is the tone positive?","Does this mention a deadline?","Is this safe to deploy?","Is more information needed?","Is the described behavior a bug?","Does the text contain a decision?"]
choice=[("Which team should handle this?",{"eng":"Engineering","design":"Design","finance":"Finance","ops":"Operations"}),("What is the main topic?",{"hw":"Hardware","sw":"Software","bus":"Business","other":"Something else"}),("What is the severity?",{"low":"Low","mid":"Medium","high":"High","crit":"Critical"}),("Which language is this written in?",{"en":"English","code":"Source code","mixed":"Mixed","other":"Other"})]
NREQ=int(sys.argv[1]) if len(sys.argv)>1 else 200
H={}
def hook(name):
    def f(mod,args):
        x=args[0].detach().reshape(-1,args[0].shape[-1]).float()
        H[name]=H.get(name,0)+x.T@x
        H[name+'#n']=H.get(name+'#n',0)+x.shape[0]
    return f
for l,layer in enumerate(d.backbone.language_model.layers):
    if hasattr(layer,'linear_attn'):
        a=layer.linear_attn
        a.in_proj_qkv.register_forward_pre_hook(hook(f'{l}.in')); a.out_proj.register_forward_pre_hook(hook(f'{l}.out'))
    else:
        a=layer.self_attn
        a.q_proj.register_forward_pre_hook(hook(f'{l}.in')); a.o_proj.register_forward_pre_hook(hook(f'{l}.out'))
    layer.mlp.gate_proj.register_forward_pre_hook(hook(f'{l}.gu')); layer.mlp.down_proj.register_forward_pre_hook(hook(f'{l}.dn'))
t0=time.time()
for i in range(NREQ):
    n=random.randint(20,260); s=random.randint(0,len(words)-n-1)
    state=' '.join(words[s:s+n]) if random.random()<0.9 else ''
    if random.random()<0.6: q={"type":"noul","instructions":random.choice(noul)}
    else:
        c=random.choice(choice); q={"type":"choice","instructions":c[0],"criteria":c[1]}
    ch=d.encoder.plan({"state":state,"questions":{"a":q}},d.max_length,d.mode)[0]
    d.reference_logits([ch])
    if i%20==0: print(i,time.time()-t0,flush=True)
torch.save(H,'hess.pt')
print('saved',time.time()-t0)
