import os
import re
import json
import sys
import time
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextGenerationPipeline,
    BitsAndBytesConfig,
)
from comet import download_model, load_from_checkpoint
from sacrebleu.metrics import CHRF
from tqdm import tqdm
import argparse

MAX_NEW_TOKENS = 256  # fixed constant
torch.backends.cuda.matmul.allow_tf32 = True  # speedup on A100

TEST_ONE_BATCH = False   # set False for full run
PRINT_N = 5             # how many examples to print for a quick spot-check

ASSISTANT_TAG_RE = re.compile(r"<\|im_start\|>\s*assistant\s*\n?", re.IGNORECASE)
USER_BLOCK_RE = re.compile(r"<\|im_start\|>\s*user.*?(?=<\|im_start\|>\s*assistant)", re.IGNORECASE | re.DOTALL)

def clean_translation(text: str, target_lang_name: str) -> str:
    """Return only the translation text (no chat tags, no prompt)."""
    if not text:
        return text

    t = text.strip()

    #    (handles cases where the model reprints the prompt before answering).
    if "<|im_start|>" in t:
        # Remove any user block(s) and keep content after the last assistant tag
        # First, strip user blocks to reduce noise
        t = USER_BLOCK_RE.sub("", t)
        # split on assistant tags and keep the tail
        parts = ASSISTANT_TAG_RE.split(t)
        t = parts[-1].strip() if parts else t

    #  look for the *last* occurrence of "<Lang>:" and keep what's after it.
    lang_label = f"{target_lang_name}:"
    idx = t.lower().rfind(lang_label.lower())
    if idx != -1:
        t = t[idx + len(lang_label):].strip()

    # Strip code fences or quotes if the model wrapped the answer
    t = t.strip("`").strip().strip('“”"\'')

    # Collapse whitespace
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def get_args():
    p = argparse.ArgumentParser(description="Translation evaluation")
    p.add_argument("langFile", type=str, help="Path to JSON with language pairs")
    p.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["bf16", "fp16", "int8", "int4"],
        help="Weight/compute mode",
    )
    p.add_argument("--batch_size", type=int, default=16)
    # ---- Added flags (default behavior unchanged) ----
    p.add_argument(
        "--direction",
        type=str,
        default="en2xx",
        choices=["en2xx", "xx2en"],
        help="Translate English→Other (en2xx) or Other→English (xx2en). Default: en2xx",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, only translate the first N usable segments after filtering.",
    )
    p.add_argument(
        "--skip_bad",
        action="store_true",
        help="If set, drop rows where is_bad_source == True (recommended).",
    )
    return p.parse_args()

LANGUAGE_BY_CODE = {
    "ar_EG": "Arabic","ar_SA": "Arabic","bg_BG": "Bulgarian","bn_BD": "Bengali","bn_IN": "Bengali",
    "ca_ES": "Catalan","cs_CZ": "Czech","da_DK": "Danish","de_DE": "German","el_GR": "Greek",
    "es_MX": "Spanish","et_EE": "Estonian","fa_IR": "Farsi","fi_FI": "Finnish","fil_PH": "Filipino",
    "fr_CA": "French","fr_FR": "French","gu_IN": "Gujarati","he_IL": "Hebrew","hi_IN": "Hindi",
    "hr_HR": "Croatian","hu_HU": "Hungarian","id_ID": "Indonesian","is_IS": "Icelandic","it_IT": "Italian",
    "ja_JP": "Japanese","kn_IN": "Kannada","ko_KR": "Korean","lt_LT": "Lithuanian","lv_LV": "Latvian",
    "ml_IN": "Malayalam","mr_IN": "Marathi","nl_NL": "Dutch","no_NO": "Norwegian","pa_IN": "Punjabi",
    "pl_PL": "Polish","pt_BR": "Portuguese","pt_PT": "Portuguese","ro_RO": "Romanian","ru_RU": "Russia",
    "sk_SK": "Slovak","sl_SI": "Slovenian","sr_RS": "Serbian","sv_SE": "Swedish","sw_KE": "Swahili",
    "sw_TZ": "Swahili","ta_IN": "Tamil","te_IN": "Telugu","th_TH": "Thai","tr_TR": "Turkish",
    "uk_UA": "Ukrainian","ur_PK": "Urdu","vi_VN": "Vietnamese","zh_CN": "Mandarin","zh_TW": "Mandarin",
    "zu_ZA": "Zulu"
}
REGION_BY_CODE = {
    "ar_EG": "Egypt","ar_SA": "Saudi Arabia","bg_BG": "Bulgaria","bn_BD": "Bangladesh","bn_IN": "India",
    "ca_ES": "Spain","cs_CZ": "Czechia","da_DK": "Denmark","de_DE": "Germany","el_GR": "Greece",
    "es_MX": "Mexico","et_EE": "Estonia","fa_IR": "Iran","fi_FI": "Finland","fil_PH": "Philippines",
    "fr_CA": "Canada","fr_FR": "France","gu_IN": "India","he_IL": "Israel","hi_IN": "India",
    "hr_HR": "Croatia","hu_HU": "Hungary","id_ID": "Indonesia","is_IS": "Iceland","it_IT": "Italy",
    "ja_JP": "Japan","kn_IN": "India","ko_KR": "South Korea","lt_LT": "Lithuania","lv_LV": "Latvia",
    "ml_IN": "India","mr_IN": "India","nl_NL": "Netherlands","no_NO": "Norway","pa_IN": "India",
    "pl_PL": "Poland","pt_BR": "Brazil","pt_PT": "Portugal","ro_RO": "Romania","ru_RU": "Russia",
    "sk_SK": "Slovakia","sl_SI": "Slovenia","sr_RS": "Serbia","sv_SE": "Sweden","sw_KE": "Kenya",
    "sw_TZ": "Tanzania","ta_IN": "India","te_IN": "India","th_TH": "Thailand","tr_TR": "Turkey",
    "uk_UA": "Ukraine","ur_PK": "Pakistan","vi_VN": "Vietnam","zh_CN": "China","zh_TW": "Taiwan",
    "zu_ZA": "South Africa"
}

model_id = "Unbabel/TowerInstruct-7B-v0.1"

def verify_model_precision(model, requested: str):
    print("\n[VERIFY] ===== Model Load Verification =====", flush=True)
    print(f"[VERIFY] Requested precision: {requested}", flush=True)
    print(f"[VERIFY] torch.cuda.is_available(): {torch.cuda.is_available()}", flush=True)
    print(f"[VERIFY] hf_device_map: {getattr(model, 'hf_device_map', None)}", flush=True)
    print(f"[VERIFY] is_loaded_in_4bit: {getattr(model, 'is_loaded_in_4bit', False)}", flush=True)
    print(f"[VERIFY] is_loaded_in_8bit: {getattr(model, 'is_loaded_in_8bit', False)}", flush=True)
    try:
        import bitsandbytes as bnb
        has_4 = any(isinstance(m, bnb.nn.Linear4bit) for m in model.modules())
        has_8 = any(isinstance(m, bnb.nn.Linear8bitLt) for m in model.modules())
        print(f"[VERIFY] Found bnb.nn.Linear4bit layers: {has_4}", flush=True)
        print(f"[VERIFY] Found bnb.nn.Linear8bitLt layers: {has_8}", flush=True)
    except Exception as e:
        print(f"[VERIFY] bitsandbytes introspection skipped: {e}", flush=True)
    try:
        name, param = next(iter(model.named_parameters()))
        print(f"[VERIFY] Example param: {name} | dtype={param.dtype} | device={param.device}", flush=True)
    except Exception as e:
        print(f"[VERIFY] Could not inspect a parameter: {e}", flush=True)
    ok = True
    if requested == "int4" and not getattr(model, "is_loaded_in_4bit", False): ok = False
    if requested == "int8" and not getattr(model, "is_loaded_in_8bit", False): ok = False
    if requested in {"fp16", "bf16"}:
        cfg_dtype = getattr(model.config, "torch_dtype", None)
        print(f"[VERIFY] model.config.torch_dtype: {cfg_dtype}", flush=True)
        if requested == "fp16" and cfg_dtype not in (torch.float16, None): ok = False
        if requested == "bf16" and cfg_dtype not in (torch.bfloat16, None): ok = False
    print(f"[VERIFY] RESULT: {'PASS' if ok else 'MISMATCH'}", flush=True)
    print("[VERIFY] ====================================\n", flush=True)

def build_pipe(precision: str) -> TextGenerationPipeline:
    print(f"[LOAD] Starting model load with --precision={precision}", flush=True)
    if precision == "int8":
        bnb = BitsAndBytesConfig(load_in_8bit=True)
        print("[LOAD] BitsAndBytesConfig: load_in_8bit=True", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", quantization_config=bnb
        )
    elif precision == "int4":
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        print("[LOAD] BitsAndBytesConfig: load_in_4bit=True, quant_type=nf4, double_quant=True, compute_dtype=float16", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", quantization_config=bnb
        )
    elif precision == "fp16":
        print("[LOAD] Loading FP16 weights (torch.float16)", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.float16
        )
    else:  # bf16
        print("[LOAD] Loading BF16 weights (torch.bfloat16)", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16
        )

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id  # needed for padded batched generate

    model.eval()
    verify_model_precision(model, precision)
    return TextGenerationPipeline(model=model, tokenizer=tok)

def translateLanguage(pipe: TextGenerationPipeline, pair: str, batch_size: int,
                      direction: str, limit: int, skip_bad: bool, precision: str):
    """
    Direction-aware translation with optional filtering/limit and filename tagging.
    Keeps original behavior for en2xx.
    """
    srcLangCode, targetLangCode = pair.split("-")

    if direction == "en2xx":
        # Original path: English → target (dataset['source'] -> dataset['target'])
        from_code, to_code = srcLangCode, targetLangCode
        to_name = LANGUAGE_BY_CODE.get(targetLangCode, targetLangCode)
        region = REGION_BY_CODE.get(targetLangCode, "")
        print(f"\n=== Translating English -> {to_name} ({region}) [pair={pair} | precision={precision}] ===\n", flush=True)
    else:
        # New path: target language → English (dataset['target'] -> dataset['source'])
        from_code, to_code = targetLangCode, srcLangCode
        from_name = LANGUAGE_BY_CODE.get(from_code, from_code)
        print(f"\n=== Translating {from_name} -> English [pair={pair} | precision={precision}] ===\n", flush=True)

    ds = load_dataset("google/wmt24pp", pair)["train"]

    # Optional filter for bad sources (recommended by WMT24++)
    if skip_bad:
        print("[FILTER] Applying is_bad_source == False", flush=True)
        ds = ds.filter(lambda ex: not bool(ex.get("is_bad_source", False)))

    # Optional cap to first N usable samples
    if limit and limit > 0:
        print(f"[LIMIT] Keeping first {min(limit, len(ds))} samples after filtering", flush=True)
        ds = ds.select(range(min(limit, len(ds))))

    # Filenames now include from-to, direction, and precision
    base = f"{from_code}-{to_code}_{direction}_{precision}"
    jsonl_path = f"translations_{base}.jsonl"
    score_path = f"scores_{base}.json"

    f_jsonl = open(jsonl_path, "w", encoding="utf-8")

    translations, srcsRefs = [], []
    B = batch_size
    num_batches = (len(ds) + B - 1) // B
    print(f"[LOOP] {num_batches} batches total (B={B}, samples={len(ds)})", flush=True)

    device = next(pipe.model.parameters()).device
    global_idx_base = 0  # to construct fallback segment ids

    for i in tqdm(range(0, len(ds), B)):
        batch = ds.select(range(i, min(i + B, len(ds))))
        # local holders just for THIS batch
        batch_srcsRefs = []
        batch_translations = []

        prompts = []
        seg_meta = []  # holds (segment_id, src, ref) aligned to prompts

        for j, ex in enumerate(batch):
            if direction == "en2xx":
                src = ex["source"]     # English input
                ref = ex["target"]     # Non-English reference (post-edit)
                to_name = LANGUAGE_BY_CODE.get(targetLangCode, targetLangCode)
                msg = f"Translate the following text from English into {to_name}.\nEnglish: {src}\n{to_name}:"
                target_lang_name_for_clean = to_name
            else:
                src = ex["target"]     # Non-English input (post-edit)
                ref = ex["source"]     # English reference
                from_name = LANGUAGE_BY_CODE.get(from_code, from_code)
                msg = f"Translate the following text into English.\n{from_name}: {src}\nEnglish:"
                target_lang_name_for_clean = "English"

            seg_id = ex.get("segment_id", ex.get("id", f"{pair}:{i + j}"))
            prompt = pipe.tokenizer.apply_chat_template(
                [{"role": "user", "content": msg}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)
            seg_meta.append((seg_id, src, ref))

            srcsRefs.append({"src": src, "ref": ref})
            batch_srcsRefs.append({"src": src, "ref": ref})

        # Tokenize and move to device
        enc = pipe.tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        input_lens = attention_mask.sum(dim=1).tolist()

        print(f"[GEN] starting batch {(i//B)+1} (size={len(batch)})", flush=True)
        t0 = time.time()
        with torch.inference_mode():
            gen_ids = pipe.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        print(f"[GEN] finished batch {(i//B)+1} in {time.time()-t0:.1f}s", flush=True)

        # Decode only the newly generated tokens and stream-write JSONL
        for j in range(gen_ids.size(0)):
            new_tokens = gen_ids[j, int(input_lens[j]):]
            raw_hyp = pipe.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            hyp = clean_translation(raw_hyp, target_lang_name_for_clean)
            translations.append(hyp)
            batch_translations.append(hyp)  # local list for this batch

            seg_id, src, ref = seg_meta[j]
            rec = {
                "segment_id": seg_id,
                "src": src,
                "hyp": hyp,
                "raw_hyp": raw_hyp,
                "ref": ref
            }
            f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")

        global_idx_base += len(batch)

        # ----- TEST BLOCK: run metrics on THIS batch only, print a few samples -----
        if TEST_ONE_BATCH:
            print("\n=== SAMPLE TRANSLATIONS (cleaned) ===")
            for k in range(min(PRINT_N, len(batch_translations))):
                print(f"[{k}] SRC: {batch_srcsRefs[k]['src']}")
                print(f"    REF: {batch_srcsRefs[k]['ref']}")
                print(f"    HYP: {batch_translations[k]}\n")

            # Compute COMET & CHRF on this batch only
            refs_b = [x["ref"] for x in batch_srcsRefs]
            comet_data_b = [{"src": x["src"], "mt": y, "ref": x["ref"]} for x, y in zip(batch_srcsRefs, batch_translations)]

            comet_path = download_model("Unbabel/wmt22-comet-da")
            comet_model = load_from_checkpoint(comet_path)
            use_gpu = 1 if torch.cuda.is_available() else 0
            comet_out_b = comet_model.predict(comet_data_b, batch_size=32, gpus=use_gpu)

            chrf = CHRF(word_order=2)
            chrf_score_b = chrf.corpus_score(batch_translations, [refs_b])

            print("\n=== BATCH-ONLY SCORES ===")
            print(f"COMET (batch):  {comet_out_b.system_score:.4f}")
            print(f"CHRF++ (batch): {chrf_score_b.score:.4f}")

            # Optionally save batch-only scores (tagged)
            out_file_b = f"scores_{base}_BATCH1.json"
            with open(out_file_b, "w", encoding="utf-8") as f:
                json.dump({
                    "pair": pair,
                    "direction": direction,
                    "precision": precision,
                    "batch_index": (i//B) + 1,
                    "COMET": comet_out_b.system_score,
                    "CHRF++": chrf_score_b.score
                }, f, ensure_ascii=False, indent=2)
            print(f"[SAVE] Wrote batch-only scores to {out_file_b}", flush=True)

            # Stop after first batch
            break
        # ----- END TEST BLOCK -----

    f_jsonl.close()
    print(f"[SAVE] Wrote per-segment outputs to {jsonl_path}", flush=True)

    # Metrics (system-level summaries)
    refs = [x["ref"] for x in srcsRefs]
    comet_data = [{"src": x["src"], "mt": y, "ref": x["ref"]} for x, y in zip(srcsRefs, translations)]

    comet_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_path)
    use_gpu = 1 if torch.cuda.is_available() else 0
    comet_out = comet_model.predict(comet_data, batch_size=32, gpus=use_gpu)

    chrf = CHRF(word_order=2)
    chrf_score = chrf.corpus_score(translations, [refs])

    out_file = score_path
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "pair": pair,
            "precision": precision,
            "samples": len(translations),
            "COMET": comet_out.system_score,
            "CHRF++": chrf_score.score
        }, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] Wrote system-level scores to {out_file}", flush=True)

def main():
    args = get_args()
    with open(args.langFile) as f:
        LANGUAGE_PAIRS = json.load(f)

    pipe = build_pipe(args.precision)

    # Tiny warm-up to JIT kernels (fast)
    print("[WARMUP] tiny generation...", flush=True)
    enc = pipe.tokenizer(
        ["<|im_start|>user Translate 'Hello' to French.<|im_end|><|im_start|>assistant"],
        return_tensors="pt", padding=True
    )
    enc = {k: v.to(next(pipe.model.parameters()).device) for k, v in enc.items()}
    with torch.inference_mode():
        _ = pipe.model.generate(**enc, max_new_tokens=8, do_sample=False)
    print("[WARMUP] done.", flush=True)

    for pair in LANGUAGE_PAIRS:
        translateLanguage(
            pipe,
            pair,
            batch_size=args.batch_size,
            direction=args.direction,
            limit=args.limit,
            skip_bad=args.skip_bad,
            precision=args.precision
        )

if __name__ == "__main__":
    main()