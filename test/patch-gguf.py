#!/usr/bin/env python3
"""
Patch the Morpheus GGUF for wllama compatibility + correct tokenization.

Two fixes applied in one pass:

1. HEAD_COUNT FIX (load compatibility)
   The llama.cpp HF converter writes mamba2.attention.head_count = 0
   (commented "unused"). wllama 3.5.1's bundled llama.cpp (dd4623a) uses
   head_count to compute the ssm_in tensor width
   (2*d_inner + 2*n_group*d_state + n_head), so 0 makes it reject the
   model. Setting head_count = 24 (= d_inner/head_dim = dt_rank) produces
   the correct width (3224) and loads in wllama while staying compatible
   with newer llama.cpp (which uses ssm.time_step_rank instead).

2. UGM TOKENIZER FIX (tokenization fidelity)
   The model was trained with a SentencePiece *unigram* tokenizer, but the
   GGUF carries tokenizer.ggml.model = "llama" which maps to llama.cpp's
   SPM tokenizer — this uses BPE-style pair merging, NOT the unigram
   Viterbi algorithm. The result: every word preceded by a space gets
   mis-tokenized (e.g. " zer" → [▁][▁][z][er] instead of [▁zer]),
   shifting the argmax and producing wrong completions.

   Fix: change model to "t5" which maps to LLAMA_VOCAB_TYPE_UGM (the real
   unigram Viterbi algorithm). Also set add_space_prefix = true (dummy
   prefix, like SP's add_dummy_prefix) and remove_extra_whitespaces = true
   (collapse spaces, like SP's normalize). These keys are read by llama.cpp
   from GGUF metadata, so no source patching or wllama fork is needed.

   After this patch, llama.cpp's tokenization matches the reference
   SentencePiece model 100%, and string-prompt completions match
   token-ID-prompt completions exactly.

Usage:
    python3 test/patch-gguf.py [input.gguf] [output.gguf]

    # Default: downloads from HF, writes model/step_0074000.Q4_K_M.ugm.gguf
    python3 test/patch-gguf.py

    # From an existing local file:
    python3 test/patch-gguf.py path/to/input.gguf path/to/output.gguf
"""
import sys, os, urllib.request

import gguf

HF_REPO = "itzune/morpheus-gguf"
HF_FILE = "morpheus-v2-mamba.Q4_K_M.gguf"
HEAD_COUNT = 24  # = d_inner / head_dim = dt_rank


def download_from_hf(dst):
    url = f"https://huggingface.co/{HF_REPO}/resolve/main/{HF_FILE}"
    print(f"Downloading from {url} …")
    urllib.request.urlretrieve(url, dst)
    print(f"  saved → {dst} ({os.path.getsize(dst):,} bytes)")


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else None
    default_dst = os.path.join(os.path.dirname(__file__), "..", "model",
                               "morpheus-v2-mamba.Q4_K_M.ugm.gguf")
    dst = sys.argv[2] if len(sys.argv) > 2 else os.path.abspath(default_dst)

    if src is None:
        # Auto-download from HF
        cache = os.path.join(os.path.dirname(__file__), "..", "model", HF_FILE)
        src = os.path.abspath(cache)
        if not os.path.exists(src):
            os.makedirs(os.path.dirname(src), exist_ok=True)
            download_from_hf(src)
        else:
            print(f"Using cached: {src}")

    print(f"Reading: {src}")
    r = gguf.GGUFReader(src)

    # ── Read ALL metadata fields (skip GGUF.* header pseudo-fields) ──
    fields = {}
    for name, field in r.fields.items():
        if name.startswith("GGUF."):
            continue
        t = field.types[0] if field.types else None
        if t == gguf.GGUFValueType.ARRAY:
            elem_t = field.types[1]
            if elem_t == gguf.GGUFValueType.STRING:
                arr = [bytes(field.parts[i]).decode("utf-8", "replace")
                       for i in field.data]
            else:
                py_type = {
                    gguf.GGUFValueType.UINT32: int,
                    gguf.GGUFValueType.INT32: int,
                    gguf.GGUFValueType.FLOAT32: float,
                    gguf.GGUFValueType.UINT64: int,
                    gguf.GGUFValueType.INT64: int,
                    gguf.GGUFValueType.FLOAT64: float,
                    gguf.GGUFValueType.BOOL: bool,
                }.get(elem_t, int)
                arr = [py_type(field.parts[i][0]) for i in field.data]
            fields[name] = ("array", elem_t, arr)
        else:
            val = field.parts[field.data[0]][0]
            if t == gguf.GGUFValueType.STRING:
                val = bytes(field.parts[field.data[0]]).decode("utf-8", "replace")
            fields[name] = ("scalar", t, val)

    # ── Apply patches ──
    arch = fields.get("general.architecture", ("scalar", None, "mamba2"))[2]
    hc_key = f"{arch}.attention.head_count"

    old_hc = fields.get(hc_key, ("scalar", None, "?"))[2]
    fields[hc_key] = ("scalar", gguf.GGUFValueType.UINT32, HEAD_COUNT)
    print(f"  {hc_key}: {old_hc} → {HEAD_COUNT}")

    old_model = fields.get("tokenizer.ggml.model", ("scalar", None, "?"))[2]
    fields["tokenizer.ggml.model"] = ("scalar", gguf.GGUFValueType.STRING, "t5")
    fields["tokenizer.ggml.add_space_prefix"] = ("scalar", gguf.GGUFValueType.BOOL, True)
    fields["tokenizer.ggml.remove_extra_whitespaces"] = ("scalar", gguf.GGUFValueType.BOOL, True)
    print(f"  tokenizer.ggml.model: {old_model!r} → 't5'  (SPM → UGM/Viterbi)")
    print(f"  tokenizer.ggml.add_space_prefix: → True")
    print(f"  tokenizer.ggml.remove_extra_whitespaces: → True")

    # ── Read tensor info ──
    tensors = []
    for t in r.tensors:
        tensors.append({
            "name": t.name,
            "shape": list(t.shape),
            "data": t.data,
            "tensor_type": t.tensor_type,
        })
    print(f"  {len(tensors)} tensors, {len(fields)} metadata fields")

    # ── Write patched GGUF ──
    print(f"Writing: {dst}")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    w = gguf.GGUFWriter(dst, arch)

    for name, (kind, t, val) in sorted(fields.items()):
        if name == "general.architecture":
            continue  # handled by GGUFWriter constructor
        if kind == "scalar":
            if t == gguf.GGUFValueType.STRING:
                w.add_string(name, val)
            elif t == gguf.GGUFValueType.UINT32:
                w.add_uint32(name, int(val))
            elif t == gguf.GGUFValueType.INT32:
                w.add_int32(name, int(val))
            elif t == gguf.GGUFValueType.FLOAT32:
                w.add_float32(name, float(val))
            elif t == gguf.GGUFValueType.UINT64:
                w.add_uint64(name, int(val))
            elif t == gguf.GGUFValueType.BOOL:
                w.add_bool(name, bool(val))
            else:
                print(f"  SKIP scalar {name}: type {t}")
        elif kind == "array":
            if t == gguf.GGUFValueType.STRING:
                w.add_array(name, [str(v) for v in val])
            else:
                w.add_array(name, list(val))

    for t in tensors:
        w.add_tensor(t["name"], t["data"], raw_dtype=t["tensor_type"])

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    # ── Verify ──
    r2 = gguf.GGUFReader(dst)

    def get_str(name):
        return bytes(r2.fields[name].parts[-1]).decode("utf-8", "replace")

    def get_u32(name):
        return int(r2.fields[name].parts[-1][0])

    def get_bool(name):
        return bool(int(r2.fields[name].parts[-1][0]))

    print(f"\nVerify:")
    print(f"  head_count        = {get_u32(f'{arch}.attention.head_count')}")
    print(f"  tokenizer model   = {get_str('tokenizer.ggml.model')}")
    print(f"  add_space_prefix  = {get_bool('tokenizer.ggml.add_space_prefix')}")
    print(f"  remove_extra_ws   = {get_bool('tokenizer.ggml.remove_extra_whitespaces')}")
    print(f"  add_bos           = {get_bool('tokenizer.ggml.add_bos_token')}")
    print(f"  add_eos           = {get_bool('tokenizer.ggml.add_eos_token')}")
    print(f"  tensors           = {len(r2.tensors)}")
    print(f"  size              = {os.path.getsize(dst):,} bytes")
    print(f"\nDone. Load this file in wllama for correct Basque completions.")


if __name__ == "__main__":
    main()
