"""Single-GPU round 12 lab coordinator, Python 3.11 stdlib."""
import argparse
import ast
import hashlib
import math
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

PKG = Path(__file__).resolve().parent
PATCH = {
    "main.py": ("49485bc7ad4e9e0b", "e898ff1da6b92e910d9a0f62c042a9b9381ce9f5603f562cf90050223a2fa8bb"),
    "spike_driven_quant/spike_omniquant.py": ("2b17e4fcfea74fce", "4f7ae925a3c1bd5104e1bbb50374f85dc351afeab8a1bc74a0f67f3d50be1ffe"),
}
BASE = {"spike_driven_quant/spike_linear.py":"857be868ad9cc740",
        "cache/dataloader_Llama_wikitext2_128.cache":"a63dd7b84d65aac6",
        "cache/testloader_Llama_c4_all.cache":"8e26a1bd520cdb50",
        "cache/testloader_Llama_wikitext2_all.cache":"ad11d68d1922048a"}
NET = "Llama-2-7b-hf"
TASKS = "piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande"
JOBS = [("r12_68m_repro",2,None), ("r12_7b_aq_s4",4,"adaptive"),
        ("r12_7b_rand_s2",2,"random"), ("r12_7b_heur_s2",2,"heuristic_down_o")]
ZS = "r12_7b_aq_s2_zs"
GATE = (157.7880401611328,90.83118438720703)
STATE = Path("log/r12_dispatch")
def check(cond, msg):
    if not cond:
        raise RuntimeError(msg)
def digest(p):
    with Path(p).open("rb") as f:
        h=hashlib.sha256()
        for block in iter(lambda:f.read(1024*1024), b""):
            h.update(block)
        return h.hexdigest()
def fingerprint():
    for p,h in BASE.items():
        check(digest(p).startswith(h), "Unexpected baseline: "+p)
    for rel,(old,new) in PATCH.items():
        check(digest(PKG/"patch"/rel)==new,"Damaged payload: "+rel)
        h=digest(rel)
        check(h==new or h.startswith(old),"Unknown installed version: "+rel)
def atomic_copy(src,dst):
    dst=Path(dst)
    dst.parent.mkdir(exist_ok=True,parents=True)
    tmp=dst.with_name(dst.name+".r12-copy")
    shutil.copyfile(src,tmp)
    os.replace(tmp,dst)
def install():
    changed=[]
    try:
        for rel,(old,new) in PATCH.items():
            if digest(rel)==new:
                continue
            backup=STATE/"before"/rel
            if not backup.exists():
                atomic_copy(rel,backup)
            check(digest(backup)==digest(rel),"Backup mismatch: "+rel)
            atomic_copy(PKG/"patch"/rel,rel)
            changed.append(rel)
            check(digest(rel)==new,"Install failed: "+rel)
    except BaseException:
        restore(changed)
        raise
    return changed
def restore(changed):
    for rel in reversed(changed):
        atomic_copy(STATE/"before"/rel,rel)
def logfile(tag):
    return max((Path("log")/tag).glob("log_rank0_*.txt"),key=lambda p:p.stat().st_mtime_ns,default=None)
def readlog(p):
    data=p.read_bytes()
    for enc in ("utf-8-sig","gb18030"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    return data.decode("utf8","replace")
def verified(tag,seed,policy):
    p=logfile(tag)
    if not p: return None
    s=readlog(p)
    ns=re.search(r"INFO Namespace\((.*)\)\s*$",s,re.M)
    if not ns: return None
    fields={k.arg:ast.literal_eval(k.value) for k in ast.parse("f("+ns[1]+")",mode="eval").body.keywords}
    want=dict(seed=seed,net="llama-68m" if policy is None else NET,
              cache_dir="./cache_seed4" if seed==4 else "./cache",
              wbits=4,abits=4,addbit=1,low_p=.9,nsamples=128,epochs=20,
              alpha=.75,let=True,lwc=True,aug_loss=True,adaptive_quota=bool(policy),
              quota_policy=policy or "adaptive",sim_mode="fake_binary_simulate",
              eval_ppl=True,orig_spikellm=False,batch_size=1,limit=-1,resume=None)
    if any(fields.get(k)!=v for k,v in want.items()): return None
    vals=[]
    for k in ("wikitext2","c4"):
        m=re.findall(r"INFO "+k+r" : ([^\s]+)",s)
        if not m: return None
        v=float(m[-1])
        check(math.isfinite(v) and v>0,tag+": invalid "+k)
        vals.append(v)
    if policy is None:
        check(all(abs(x-y)<1e-9 for x,y in zip(vals,GATE)),f"Gate mismatch: {vals}, expected {GATE}")
    else:
        rows=re.findall(r"\[("+re.escape(policy)+r"_quota)\] block MSE .*?high_MACs (\d+) -> (\d+)",s)
        check(len(rows)==32 and all(int(a)==20253696 and 0<int(b)<=int(a) for _,a,b in rows),tag+": quota audit failed")
    return vals
def invoke(tag,argv):
    folder=Path("log")/tag
    folder.mkdir(exist_ok=True,parents=True)
    print("Starting",tag,flush=True)
    with (folder/"r12_console.txt").open("ab") as f:
        subprocess.run([sys.executable,"main.py",*argv,"--output_dir","./log/"+tag],stdout=f,stderr=subprocess.STDOUT,check=True)
def arguments(model,net,seed,epochs):
    return ["--model",str(model),"--net",net,"--wbits","4","--abits","4","--addbit","1","--low_p","0.9",
            "--let","--lwc","--aug_loss","--alpha","0.75","--let_lr","1e-3","--lwc_lr","1e-2","--nsamples","128",
            "--epochs",str(epochs),"--seed",str(seed),"--cache_dir","./cache_seed4" if seed==4 else "./cache"]
def result(msg):
    print(msg,flush=True)
    with (STATE/"result.txt").open("a",encoding="utf-8") as f: print(msg,file=f)
def task_result():
    p=logfile(ZS)
    if not p: return None
    for line in reversed(readlog(p).splitlines()):
        if "INFO {" not in line or "'results':" not in line: continue
        metrics={}
        for name in TASKS.split(","):
            m=re.search(re.escape(repr(name))+r":\s*(\{[^{}]*\})",line)
            if not m: return None
            row=ast.literal_eval(m[1])
            if not isinstance(row.get("acc"),(int,float)) or not 0<=row["acc"]<=1: return None
            metrics[name]={k:v for k,v in row.items() if k in ("acc","acc_norm")}
        return metrics
def export():
    out=Path("lab_round12_return_codex.zip")
    tmp=out.with_suffix(".partial")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED) as z:
        for folder in [STATE,*[Path("log")/t for t,_,_ in JOBS],Path("log")/ZS]:
            for p in folder.glob("*"):
                if p.is_file() and p.suffix in (".txt",".log",".failed"):
                    z.write(p,p.as_posix())
    os.replace(tmp,out)
def run(mode):
    model=Path("models",NET).resolve()
    fingerprint()
    check(model.is_dir(),"Missing "+str(model))
    for n in ("llama-68m",NET):
        for t in ("act_scales","act_shifts"):
            check(Path(t,n+".pt").is_file(),"Missing "+t+"/"+n+".pt")
    subprocess.run([sys.executable,"-c","import sys,torch,transformers; print('Visible GPUs:',torch.cuda.device_count(),flush=True); sys.exit(0 if torch.cuda.device_count()==1 else 1)"],check=True)
    check(Path("env_report.py").is_file(),"Missing env_report.py")
    check(Path("cache_seed4/dataloader_Llama_wikitext2_128.cache").is_file(),"Restore the historical Full seed4 calibration cache before running.")
    for name in ("testloader_Llama_c4_all.cache","testloader_Llama_wikitext2_all.cache"):
        dst=Path("cache_seed4",name)
        if dst.exists(): check(digest(dst)==digest(Path("cache",name)),"Mismatched evaluation cache: "+name)
        elif mode!="preflight": atomic_copy(Path("cache",name),dst)
    if mode=="preflight":
        print("Preflight PASS"); return
    changed=install()
    try:
        subprocess.run([sys.executable,"env_report.py","--out",str(STATE/"environment.txt")],check=True)
        result("GPU selector "+os.environ["CUDA_VISIBLE_DEVICES"]+" | seed4 calibration SHA256 "+digest("cache_seed4/dataloader_Llama_wikitext2_128.cache"))
        for tag,seed,policy in JOBS:
            values=verified(tag,seed,policy)
            if not values:
                argv=arguments(model if policy else "JackFram/llama-68m",NET if policy else "llama-68m",seed,20)
                if policy: argv+=["--adaptive_quota","--quota_policy",policy]
                invoke(tag,argv+["--eval_ppl"])
                values=verified(tag,seed,policy)
                check(values,tag+": missing valid result")
            result(f"{tag} WT2={values[0]} C4={values[1]}")
            if policy is None:
                changed=[]  # patch accepted
                if mode=="gate": return
    except BaseException:
        restore(changed)
        raise
    ck=Path("log/r11_7b_aq_s2/omni_parameters.pth")
    if not ck.exists():
        result("SKIPPED zero-shot: missing "+str(ck)); return
    metrics=task_result()
    if not metrics:
        # Older checkpoints may be upgraded in place by main.py. Evaluate on a private copy.
        copy=STATE/"zs_resume.pth"
        atomic_copy(ck,copy)
        invoke(ZS,arguments(model,NET,2,0)+["--adaptive_quota","--resume",str(copy),"--tasks",TASKS,"--num_fewshot","0"])
        metrics=task_result()
        check(metrics,"Six-task metrics incomplete")
    result("zero-shot "+str(metrics))
if __name__=="__main__":
    import msvcrt
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["run","gate","preflight"])
    mode=parser.parse_args().mode
    STATE.mkdir(parents=True,exist_ok=True)
    with (STATE/"dispatch.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        try:
            result(time.strftime("%Y-%m-%d %H:%M:%S")+" "+mode)
            run(mode)
        except BaseException as exc:
            result("STOP "+repr(exc)); raise
        finally:
            export()
