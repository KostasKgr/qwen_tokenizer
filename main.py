import os
import sys

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_VERBOSITY"] = "error"

from transformers import AutoTokenizer

t = AutoTokenizer.from_pretrained("Qwen/Qwen3.8-27B")

paths = [
    "/home/myuser/projects/myproj",
    "/projects/myproj",
    "/project/myproj",
    "/proj/myproj",
    "/p/myproj",
    "/opt/myproj",
    "/work/myproj",
    "/src/myproj",
    "/repos/myrepo",
]

inputs = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else paths

for s in inputs:
    ids = t.encode(s, add_special_tokens=False)
    pieces = [t.decode([x]) for x in ids]
    print(f"{len(ids):2}  {s:32} {pieces}")
