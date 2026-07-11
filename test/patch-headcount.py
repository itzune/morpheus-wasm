#!/usr/bin/env python3
"""
Patch mamba2.attention.head_count from 0 → 24 in the Morpheus GGUF.

WHY
---
The llama.cpp HF converter writes `mamba2.attention.head_count = 0` (commented
"unused, but seemingly required when loading"). The dd4623a-era llama.cpp
loader — which wllama 3.5.1 bundles — computes the ssm_in projection width as:

        d_in_proj = 2*d_inner + 2*n_group*d_state + n_head

reading n_head from `attention.head_count` (=0), giving d_in_proj=3200 and
rejecting the actual tensor (3224 = 3200 + dt_rank). The result is a hard load
failure in every released wllama (3.2.3–3.5.1):

        tensor 'blk.0.ssm_in.weight' has wrong shape; expected: {768, 3200}
        got: {768, 3224}

Newer llama.cpp (post 07-2025) fixes this by using `ssm.time_step_rank` (=24)
instead of head_count, so it loads fine — but wllama is pinned to the old build.

THE FIX
-------
Set head_count = 24 (= d_inner / head_dim = 1536/64 = dt_rank). This makes the
dd4623a loader compute d_in_proj = 3224 (matching the tensor) AND is harmless
for newer loaders (which use dt_rank and ignore head_count). A 4-byte in-place
patch of the uint32 value; file size is unchanged.

USAGE
-----
    python3 test/patch-headcount.py
    # → downloads the upstream GGUF from HF (if missing) and writes
    #   model/morpheus-v2-mamba.Q4_K_M.wllama.gguf

Requires the `gguf` package (pip install gguf, or use the llama.cpp venv).
"""
import os, sys, struct, shutil, urllib.request

# Resolve repo root (parent of test/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, 'model', 'morpheus-v2-mamba.Q4_K_M.gguf')
DST = os.path.join(ROOT, 'model', 'morpheus-v2-mamba.Q4_K_M.wllama.gguf')
HF_URL = 'https://huggingface.co/itzune/morpheus-gguf/resolve/main/morpheus-v2-mamba.Q4_K_M.gguf'
NEW_HEAD_COUNT = 24  # = d_inner / head_dim = 1536 / 64 = dt_rank

import gguf  # noqa: E402

# Download the upstream GGUF if we don't have it locally.
if not os.path.exists(SRC):
    os.makedirs(os.path.dirname(SRC), exist_ok=True)
    print(f'downloading upstream GGUF from HF → {SRC}')
    urllib.request.urlretrieve(HF_URL, SRC)
    print(f'  done ({os.path.getsize(SRC) // 1_048_576} MB)')

r = gguf.GGUFReader(SRC)
field = r.fields.get('mamba2.attention.head_count')
assert field is not None, 'mamba2.attention.head_count field not found — not a Mamba-2 GGUF?'

# GGUF value layout: [name_len:u64][name:u8*N][type:u32][value:u32]
name_len = int(field.parts[0][0])
name_bytes = bytes(field.parts[1])
type_enum = int(field.parts[2][0])
old_val = int(field.parts[3][0])
value_offset = field.offset + 8 + name_len + 4

print(f'field: {name_bytes.decode()}')
print(f'  offset={field.offset}  value_offset={value_offset}  type={type_enum} (4=UINT32)  value={old_val}')
assert name_bytes.decode() == 'mamba2.attention.head_count'
assert type_enum == 4, f'expected UINT32 (4), got {type_enum}'

if old_val == NEW_HEAD_COUNT:
    print(f'already {NEW_HEAD_COUNT} — copying as-is')
    shutil.copyfile(SRC, DST)
    sys.exit(0)

shutil.copyfile(SRC, DST)
with open(DST, 'r+b') as f:
    f.seek(value_offset)
    f.write(struct.pack('<I', NEW_HEAD_COUNT))
print(f'patched head_count {old_val} → {NEW_HEAD_COUNT}  ({DST})')

# Verify.
r2 = gguf.GGUFReader(DST)
v = int(r2.fields['mamba2.attention.head_count'].parts[3][0])
assert v == NEW_HEAD_COUNT, f'verify failed: got {v}'
print(f'verify: head_count = {v}  ✓')
