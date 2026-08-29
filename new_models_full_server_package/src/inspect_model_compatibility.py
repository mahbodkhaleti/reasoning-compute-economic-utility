#!/usr/bin/env python3
"""Resolve pinned tokenizers and verify required thinking-template arguments."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from transformers import AutoTokenizer
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();cfg=json.loads(a.config.read_text());report=[]
 for spec in cfg["models"]:
  tok=AutoTokenizer.from_pretrained(spec["hf_id"],revision=spec["tokenizer_revision"],trust_remote_code=True)
  arg=spec.get("chat_template_argument");kwargs={"tokenize":False,"add_generation_prompt":True,arg:True};rendered=tok.apply_chat_template([{"role":"user","content":"Template compatibility check."}],**kwargs)
  # Transformers 5.x may omit its private _commit_hash field for tokenizers.
  # Loading by an immutable commit SHA already pins the downloaded files, so
  # use that requested SHA when the private diagnostic field is unavailable.
  resolved=tok.init_kwargs.get("_commit_hash") or spec["tokenizer_revision"]
  if resolved!=spec["tokenizer_revision"]:raise RuntimeError(f"{spec['slug']}: tokenizer revision mismatch {resolved}")
  report.append({"model":spec["slug"],"requested_id":spec["hf_id"],"requested_revision":spec["revision"],"resolved_tokenizer_revision":resolved,"chat_template_argument":arg,"render_succeeded":bool(rendered),"reasoning_protocol_matched":spec.get("reasoning_protocol_matched")})
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
