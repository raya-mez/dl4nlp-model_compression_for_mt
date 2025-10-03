import os
import re
import json
import time
import torch
import logging
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

MAX_NEW_TOKENS = 256  # fixed constant
torch.backends.cuda.matmul.allow_tf32 = True  # speedup on A100

TEST_ONE_BATCH = False   # set False for full run
PRINT_N = 5             # how many examples to print for a quick spot-check

ASSISTANT_TAG_RE = re.compile(r"\[\/INST\]", re.IGNORECASE)
USER_BLOCK_RE = re.compile(r"<s>\s*\[INST\].*?\[\/INST\]", re.IGNORECASE | re.DOTALL)

def clean_translation(text: str, target_lang_name: str) -> str:
    """Return only the translation text (no chat tags, no prompt)."""
    if not text:
        return text

    t = text.strip()

    #    (handles cases where the model reprints the prompt before answering).
    if "[INST]" in t:
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
    p.add_argument(
        '--lang_pairs', 
        type=str, nargs='+', 
        default=["en-fr_FR", "en-de_DE", "en-it_IT", "en-hi_IN", "en-sw_KE", "en-th_TH"]
    )
    p.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "int8", "int4"],
        help="Weight/compute mode",
    )
    p.add_argument("--batch_size", type=int, default=16)
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

model_id = "meta-llama/Llama-2-7b-chat-hf"

def verify_model_precision(model, requested: str):
    logging.info("\n[VERIFY] ===== Model Load Verification =====")
    logging.info(f"[VERIFY] Requested precision: {requested}")
    logging.info(f"[VERIFY] torch.cuda.is_available(): {torch.cuda.is_available()}")
    logging.info(f"[VERIFY] hf_device_map: {getattr(model, 'hf_device_map', None)}")
    logging.info(f"[VERIFY] is_loaded_in_4bit: {getattr(model, 'is_loaded_in_4bit', False)}")
    logging.info(f"[VERIFY] is_loaded_in_8bit: {getattr(model, 'is_loaded_in_8bit', False)}")
    try:
        import bitsandbytes as bnb
        has_4 = any(isinstance(m, bnb.nn.Linear4bit) for m in model.modules())
        has_8 = any(isinstance(m, bnb.nn.Linear8bitLt) for m in model.modules())
        logging.info(f"[VERIFY] Found bnb.nn.Linear4bit layers: {has_4}")
        logging.info(f"[VERIFY] Found bnb.nn.Linear8bitLt layers: {has_8}")
    except Exception as e:
        logging.info(f"[VERIFY] bitsandbytes introspection skipped: {e}")
    try:
        name, param = next(iter(model.named_parameters()))
        logging.info(f"[VERIFY] Example param: {name} | dtype={param.dtype} | device={param.device}")
    except Exception as e:
        logging.info(f"[VERIFY] Could not inspect a parameter: {e}")
    ok = True
    if requested == "int4" and not getattr(model, "is_loaded_in_4bit", False): ok = False
    if requested == "int8" and not getattr(model, "is_loaded_in_8bit", False): ok = False
    if requested in {"fp16", "bf16"}:
        cfg_dtype = getattr(model.config, "torch_dtype", None)
        logging.info(f"[VERIFY] model.config.torch_dtype: {cfg_dtype}")
        if requested == "fp16" and cfg_dtype not in (torch.float16, None): ok = False
        if requested == "bf16" and cfg_dtype not in (torch.bfloat16, None): ok = False
    logging.info(f"[VERIFY] RESULT: {'PASS' if ok else 'MISMATCH'}")
    logging.info("[VERIFY] ====================================\n")

def build_pipe(precision: str) -> TextGenerationPipeline:
    logging.info(f"[LOAD] Starting model load with --precision={precision}")
    if precision == "int8":
        bnb = BitsAndBytesConfig(load_in_8bit=True)
        logging.info("[LOAD] BitsAndBytesConfig: load_in_8bit=True")
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
        logging.info("[LOAD] BitsAndBytesConfig: load_in_4bit=True, quant_type=nf4, double_quant=True, compute_dtype=float16")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", quantization_config=bnb
        )
    elif precision == "fp16":
        logging.info("[LOAD] Loading FP16 weights (torch.float16)")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.float16
        )
    else:  # bf16
        logging.info("[LOAD] Loading BF16 weights (torch.bfloat16)")
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16
        )

    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id  # needed for padded batched generate

    model.eval()
    verify_model_precision(model, precision)
    return TextGenerationPipeline(model=model, tokenizer=tok)

def translateLanguage(pipe: TextGenerationPipeline, pair: str, batch_size: int):
    srcLangCode, targetLangCode = pair.split("-")
    targetLangName = LANGUAGE_BY_CODE.get(targetLangCode, targetLangCode)
    targetRegion = REGION_BY_CODE.get(targetLangCode, "")
    logging.info(f"\n=== Translating English -> {targetLangName} ({targetRegion}) ===\n")

    ds = load_dataset("google/wmt24pp", pair)["train"]

    # JSONL file to store per-segment outputs
    jsonl_path = f"translations_{targetLangCode}.jsonl"
    f_jsonl = open(jsonl_path, "w", encoding="utf-8")

    translations, srcsRefs = [], []
    B = batch_size
    num_batches = (len(ds) + B - 1) // B
    logging.info(f"[LOOP] {num_batches} batches total (B={B}, samples={len(ds)})")

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
            src = ex["source"]; ref = ex["target"]
            seg_id = ex.get("segment_id", ex.get("id", f"{pair}:{i + j}"))

            msg = f"Translate the following text from English into {targetLangName}.\nEnglish: {src}\n{targetLangName}:"
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

        logging.info(f"\n[GEN] starting batch {(i//B)+1} (size={len(batch)})")
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
        logging.info(f"[GEN] finished batch {(i//B)+1} in {time.time()-t0:.1f}s")

        # Decode only the newly generated tokens and stream-write JSONL
        for j in range(gen_ids.size(0)):
            new_tokens = gen_ids[j, int(input_lens[j]):]
            raw_hyp = pipe.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            hyp = clean_translation(raw_hyp, targetLangName)
            translations.append(hyp)
            batch_translations.append(hyp)  # local list for this batch

            seg_id, src, ref = seg_meta[j]
            rec = {"segment_id": seg_id, "src": src, "hyp": hyp, "raw_hyp": raw_hyp, "ref": ref}
            f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")

        global_idx_base += len(batch)

        # ----- TEST BLOCK: run metrics on THIS batch only, print a few samples -----
        if TEST_ONE_BATCH:
            logging.info("\n=== SAMPLE TRANSLATIONS (cleaned) ===")
            for k in range(min(PRINT_N, len(batch_translations))):
                logging.info(f"[{k}] SRC: {batch_srcsRefs[k]['src']}")
                logging.info(f"    REF: {batch_srcsRefs[k]['ref']}")
                logging.info(f"    HYP: {batch_translations[k]}\n")

            # Compute COMET & CHRF on this batch only
            refs_b = [x["ref"] for x in batch_srcsRefs]
            comet_data_b = [{"src": x["src"], "mt": y, "ref": x["ref"]} for x, y in zip(batch_srcsRefs, batch_translations)]

            comet_path = download_model("Unbabel/wmt22-comet-da")
            comet_model = load_from_checkpoint(comet_path)
            use_gpu = 1 if torch.cuda.is_available() else 0
            comet_out_b = comet_model.predict(comet_data_b, batch_size=32, gpus=use_gpu)

            chrf = CHRF(word_order=2)
            chrf_score_b = chrf.corpus_score(batch_translations, [refs_b])

            logging.info("\n=== BATCH-ONLY SCORES ===")
            logging.info(f"COMET (batch):  {comet_out_b.system_score:.4f}")
            logging.info(f"CHRF++ (batch): {chrf_score_b.score:.4f}")

            # Optionally save batch-only scores
            out_file_b = f"scores_{targetLangCode}_BATCH1.json"
            with open(out_file_b, "w", encoding="utf-8") as f:
                json.dump({
                    "language": targetLangCode,
                    "batch_index": (i//B) + 1,
                    "COMET": comet_out_b.system_score,
                    "CHRF++": chrf_score_b.score
                }, f, ensure_ascii=False, indent=2)
            logging.info(f"[SAVE] Wrote batch-only scores to {out_file_b}")

            # Stop after first batch
            break
        # ----- END TEST BLOCK -----


    f_jsonl.close()
    logging.info(f"[SAVE] Wrote per-segment outputs to {jsonl_path}")

    # Metrics (system-level summaries)
    refs = [x["ref"] for x in srcsRefs]
    comet_data = [{"src": x["src"], "mt": y, "ref": x["ref"]} for x, y in zip(srcsRefs, translations)]

    comet_path = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_path)
    use_gpu = 1 if torch.cuda.is_available() else 0
    comet_out = comet_model.predict(comet_data, batch_size=32, gpus=use_gpu)

    chrf = CHRF(word_order=2)
    chrf_score = chrf.corpus_score(translations, [refs])

    out_file = f"scores_{targetLangCode}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "language": targetLangCode,
            "COMET": comet_out.system_score,
            "CHRF++": chrf_score.score
        }, f, ensure_ascii=False, indent=2)
    logging.info(f"[SAVE] Wrote system-level scores to {out_file}")

def main():
    args = get_args()
    LANGUAGE_PAIRS = args.lang_pairs

    pipe = build_pipe(args.precision)

    # Tiny warm-up to JIT kernels (fast)
    logging.info("[WARMUP] tiny generation...")
    warmup_msg = (
        "Translate the following text from English into French.\n"
        "English: Hello\n"
        "French:"
    )
    warmup_prompt = pipe.tokenizer.apply_chat_template(
        [{"role": "user", "content": warmup_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )
    device = next(pipe.model.parameters()).device
    warmup_inputs = pipe.tokenizer([warmup_prompt], return_tensors="pt", padding=True)
    warmup_inputs = {k: v.to(device) for k, v in warmup_inputs.items()}
    with torch.inference_mode():
        warmup_seqs = pipe.model.generate(
            **warmup_inputs,
            max_new_tokens=24,
            do_sample=False,
        )
    input_len = int(warmup_inputs["attention_mask"].sum(dim=1).tolist()[0])
    new_tokens = warmup_seqs[0, input_len:]
    decoded_output = pipe.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    logging.info("[WARMUP] done.")
    logging.info(f"[WARMUP OUTPUT] {clean_translation(decoded_output, 'French')}")

    for pair in LANGUAGE_PAIRS:
        translateLanguage(pipe, pair, batch_size=args.batch_size)

if __name__ == "__main__":
    main()