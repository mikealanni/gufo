import json, random, sqlite3, os
import pyarrow.parquet as pq
random.seed(11)
BOS="<|startoftext|>"
def prompt(text): return f"{BOS}<|im_start|>user\n{text[:1200]}<|im_end|>\n<|im_start|>assistant\n"
t=pq.read_table("uc/data/train_sft-00000-of-00003-a3ecf92756993583.parquet").to_pylist()
random.shuffle(t)
docs=[]
for r in t[:4600]:
    u=[m["content"] for m in r["messages"] if m["role"]=="user"]
    if u: docs.append(prompt(u[0]))
p=os.path.expanduser("~/.local/share/opencode/opencode.db")
c=sqlite3.connect(f"file:{p}?mode=ro",uri=True)
n=0
for (mid,data) in c.execute("select id,data from message order by random() limit 6000"):
    if json.loads(data).get("role")!="user": continue
    texts=[]
    for (pd,) in c.execute("select data from part where message_id=? order by time_created",(mid,)):
        j=json.loads(pd)
        if j.get("type")=="text" and j.get("text"): texts.append(j["text"])
    tx="\n".join(texts).strip()
    if len(tx)>30:
        docs.append(prompt(tx)); n+=1
    if n>=1400: break
random.shuffle(docs)
open("gen_prompts.docs","w").write("".join(d+"\n=====DOC=====\n" for d in docs))
print(len(docs))
