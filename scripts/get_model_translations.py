import os
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import time
import logging
import torch
from datasets import load_dataset
from transformers import TextGenerationPipeline

from model import load_model_and_tokenizer, verify_model_precision
from utils import clean_translation
from compute_scores import compute_corpus_level_scores
from constants import LANGUAGE_BY_CODE, REGION_BY_CODE, MAX_NEW_TOKENS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

torch.backends.cuda.matmul.allow_tf32 = True


def build_pipe(model_id_key, precision, kbit_load_dir=None):
    if model_id_key != 'pseudo_quant':
        model, tokenizer = load_model_and_tokenizer(model_id_key, precision)
        verify_model_precision(model, precision)
    else:
        model, tokenizer = load_model_and_tokenizer(model_id_key, precision, kbit_load_dir=kbit_load_dir)
    model.eval()
    return TextGenerationPipeline(model=model, tokenizer=tokenizer)


def translate_language(pipe, pair, result_base_dir, batch_size, direction, limit, skip_bad, test_one_batch=False,
                       num_examples_print=5):
    path = Path(result_base_dir)
    parts = path.parts

    if parts[-2] == 'pseudo_quant':
        precision = parts[-1].split('_')[0]
        src_lang_code, target_lang_code = pair.split("-")
        base_save_suffix = f"{target_lang_code}"
        source_lang_name = 'English'
        target_lang_name = LANGUAGE_BY_CODE.get(target_lang_code, target_lang_code)
        target_region = REGION_BY_CODE.get(target_lang_code, "")
        logging.info(f"\n=== Translating English -> {target_lang_name} ({target_region}) ===\n")
    elif parts[-2] == 'llama':
        precision = parts[-1]
        src_lang_code, target_lang_code = pair.split("-")
        base_save_suffix = f"{target_lang_code}_{direction}_{precision}"
        source_lang_name = 'English'
        target_lang_name = LANGUAGE_BY_CODE.get(target_lang_code, target_lang_code)
        target_region = REGION_BY_CODE.get(target_lang_code, "")
        logging.info(f"\n=== Translating English -> {target_lang_name} ({target_region}) ===\n")
    else:
        precision = parts[-1]
        if direction == 'en2xx':
            src_lang_code, target_lang_code = pair.split("-")
            base_save_suffix = f"{target_lang_code}_{direction}_{precision}"
            source_lang_name = 'English'
            target_lang_name = LANGUAGE_BY_CODE.get(target_lang_code, target_lang_code)
            target_region = REGION_BY_CODE.get(target_lang_code, "")
            logging.info(f"\n=== Translating English -> {target_lang_name} ({target_region}) ===\n")
        else:
            target_lang_code, src_lang_code = pair.split("-")
            base_save_suffix = f"{src_lang_code}_{direction}_{precision}"
            source_lang_name = LANGUAGE_BY_CODE.get(src_lang_code, src_lang_code)
            target_lang_name = 'English'
            target_region = 'England'
            logging.info(f"\n=== Translating English -> {target_lang_name} ({target_region}) ===\n")

    ds = load_dataset("google/wmt24pp", pair)["train"]

    if skip_bad:
        print("[FILTER] Applying is_bad_source == False", flush=True)
        ds = ds.filter(lambda record: not bool(record.get("is_bad_source", False)))

    if limit and limit > 0:
        print(f"[LIMIT] Keeping first {min(limit, len(ds))} samples after filtering", flush=True)
        ds = ds.select(range(min(limit, len(ds))))

    # JSONL file to store per-segment outputs
    jsonl_path = os.path.join(result_base_dir, f"translations_{base_save_suffix}.jsonl")
    f_jsonl = open(jsonl_path, "w", encoding="utf-8")

    translations, srcs_refs = [], []
    num_batches = (len(ds) + batch_size - 1) // batch_size
    logging.info(f"[LOOP] {num_batches} batches total (B={batch_size}, samples={len(ds)})")

    device = next(pipe.model.parameters()).device
    global_idx_base = 0  # to construct fallback segment ids

    for i in tqdm(range(0, len(ds), batch_size)):
        batch = ds.select(range(i, min(i + batch_size, len(ds))))
        # local holders just for THIS batch
        batch_srcs_refs = []
        batch_translations = []

        prompts = []
        seg_meta = []  # holds (segment_id, src, ref) aligned to prompts

        for j, ex in enumerate(batch):
            if direction == 'en2xx':
                src = ex["source"]
                ref = ex["target"]
                seg_id = ex.get("segment_id", ex.get("id", f"{pair}:{i + j}"))

                msg = f"Translate the following text from English into {target_lang_name}.\nEnglish: {src}\n{target_lang_name}:"
            else:
                src = ex["target"]
                ref = ex["source"]
                seg_id = ex.get("segment_id", ex.get("id", f"{pair}:{i + j}"))

                msg = f"Translate the following text into English.\n{source_lang_name}: {src}\nEnglish:"

            prompt = pipe.tokenizer.apply_chat_template(
                [{"role": "user", "content": msg}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)
            seg_meta.append((seg_id, src, ref))

            srcs_refs.append({"src": src, "ref": ref})
            batch_srcs_refs.append({"src": src, "ref": ref})

        # Tokenize and move to device
        enc = pipe.tokenizer(prompts, return_tensors='pt', padding=True, truncation=False)
        input_ids = enc['input_ids'].to(device)
        attention_mask = enc['attention_mask'].to(device)
        input_lens = attention_mask.sum(dim=1).tolist()

        logging.info(f"\n[GEN] starting batch {(i//batch_size)+1} (size={len(batch)})")
        start_time = time.time()
        with torch.inference_mode():
            gen_ids = pipe.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        logging.info(f"[GEN] finished batch {(i//batch_size)+1} in {time.time()-start_time:.1f}s")

        # Decode only the newly generated tokens and stream-write JSONL
        for j in range(gen_ids.size(0)):
            new_tokens = gen_ids[j, int(input_lens[j]):]
            raw_hyp = pipe.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            if direction == 'en2xx':
                hyp = clean_translation(raw_hyp, target_lang_name)
            else:
                hyp = clean_translation(raw_hyp, source_lang_name)
            translations.append(hyp)
            batch_translations.append(hyp)  # local list for this batch

            seg_id, src, ref = seg_meta[j]
            rec = {"segment_id": seg_id, "src": src, "hyp": hyp, "raw_hyp": raw_hyp, "ref": ref}
            f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")

        global_idx_base += len(batch)

        # ----- TEST BLOCK: run metrics on THIS batch only, print a few samples -----
        if test_one_batch:
            logging.info("\n=== SAMPLE TRANSLATIONS (cleaned) ===")
            for k in range(min(num_examples_print, len(batch_translations))):
                logging.info(f"[{k}] SRC: {batch_srcs_refs[k]['src']}")
                logging.info(f"REF: {batch_srcs_refs[k]['ref']}")
                logging.info(f"HYP: {batch_translations[k]}\n")

            # Compute COMET & CHRF on this batch only
            srcs_b = [x["src"] for x in batch_srcs_refs]
            refs_b = [x["ref"] for x in batch_srcs_refs]

            batch_scores = compute_corpus_level_scores(batch_translations, refs_b, sources=srcs_b, include_comet=True)

            logging.info("\n=== BATCH-ONLY SCORES ===")
            logging.info(f"COMET (batch):  {batch_scores['COMET_Corpus'].system_score:.4f}")
            logging.info(f"CHRF++ (batch): {batch_scores['CHRF_Corpus'].score:.4f}")
            logging.info(f"BLEU (batch): {batch_scores['BLEU_Corpus'].score:.4f}")

            # Optionally save batch-only scores
            out_file_b = os.path.join(result_base_dir, f"scores_{base_save_suffix}_BATCH1.json")
            with open(out_file_b, "w", encoding="utf-8") as f:
                json.dump({
                    "language": target_lang_code,
                    "batch_index": (i//batch_size) + 1,
                    "COMET": batch_scores['COMET_Corpus'].system_score,
                    "CHRF++": batch_scores['CHRF_Corpus'].score,
                    "BLEU": batch_scores['BLEU_Corpus'].score
                }, f, ensure_ascii=False, indent=2)
            logging.info(f"[SAVE] Wrote batch-only scores to {out_file_b}")

            # Stop after first batch
            break

    f_jsonl.close()
    logging.info(f"[SAVE] Wrote per-segment outputs to {jsonl_path}")

    # Metrics (system-level summaries)
    srcs = [x["src"] for x in srcs_refs]
    refs = [x["ref"] for x in srcs_refs]
    full_scores = compute_corpus_level_scores(translations, refs, sources=srcs, include_comet=True)

    out_file = os.path.join(result_base_dir, f"scores_{base_save_suffix}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "language": target_lang_code,
            "COMET": full_scores['COMET_Corpus'].system_score,
            "CHRF++": full_scores['CHRF_Corpus'].score,
            "BLEU": full_scores['BLEU_Corpus'].score
        }, f, ensure_ascii=False, indent=2)
    logging.info(f"[SAVE] Wrote system-level scores to {out_file}")


def main(arguments):
    if arguments.model_id_key != 'pseudo_quant':
        os.makedirs(os.path.join(arguments.result_base_dir, arguments.model_id_key.split('_')[0].strip()), exist_ok=True)
        os.makedirs(os.path.join(arguments.result_base_dir, arguments.model_id_key.split('_')[0].strip(),
                                 arguments.precision), exist_ok=True)

        final_result_save_base_dir = os.path.join(arguments.result_base_dir, arguments.model_id_key.split('_')[0].strip(),
                                                  arguments.precision)

        language_pairs = arguments.lang_pairs

        pipe = build_pipe(args.model_id_key, args.precision)

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
            add_generation_prompt=True)
        device = next(pipe.model.parameters()).device
        warmup_inputs = pipe.tokenizer([warmup_prompt], return_tensors='pt', padding=True)
        warmup_inputs = {k: v.to(device) for k, v in warmup_inputs.items()}
        with torch.inference_mode():
            warmup_seqs = pipe.model.generate(
                **warmup_inputs,
                max_new_tokens=24,
                do_sample=False,
            )
        input_len = int(warmup_inputs['attention_mask'].sum(dim=1).tolist()[0])
        new_tokens = warmup_seqs[0, input_len:]
        decoded_output = pipe.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        logging.info("[WARMUP] done.")
        logging.info(f"[WARMUP OUTPUT] {clean_translation(decoded_output, 'French')}")

        for pair in language_pairs:
            translate_language(pipe, pair, final_result_save_base_dir, arguments.batch_size, arguments.direction,
                               arguments.limit, arguments.skip_bad, test_one_batch=arguments.test_one_batch,
                               num_examples_print=arguments.num_examples_print)
    else:
        for pseudo_quant_bit in arguments.pseudo_quant_precision:
            os.makedirs(os.path.join(arguments.result_base_dir, arguments.model_id_key), exist_ok=True)
            os.makedirs(os.path.join(arguments.result_base_dir, arguments.model_id_key,
                                     f'{pseudo_quant_bit:02d}_bit_model'), exist_ok=True)

            final_result_save_base_dir = os.path.join(arguments.result_base_dir,
                                                      arguments.model_id_key, f'{pseudo_quant_bit:02d}_bit_model')

            language_pairs = arguments.lang_pairs

            pipe = build_pipe(args.model_id_key, pseudo_quant_bit, kbit_load_dir=os.path.join(
                arguments.result_base_dir, 'pseudo_quant_models', f'{pseudo_quant_bit:02d}_bit_model'))

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
                add_generation_prompt=True)
            device = next(pipe.model.parameters()).device
            warmup_inputs = pipe.tokenizer([warmup_prompt], return_tensors='pt', padding=True)
            warmup_inputs = {k: v.to(device) for k, v in warmup_inputs.items()}
            with torch.inference_mode():
                warmup_seqs = pipe.model.generate(
                    **warmup_inputs,
                    max_new_tokens=24,
                    do_sample=False,
                )
            input_len = int(warmup_inputs['attention_mask'].sum(dim=1).tolist()[0])
            new_tokens = warmup_seqs[0, input_len:]
            decoded_output = pipe.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            logging.info("[WARMUP] done.")
            logging.info(f"[WARMUP OUTPUT] {clean_translation(decoded_output, 'French')}")

            for pair in language_pairs:
                translate_language(pipe, pair, final_result_save_base_dir, arguments.batch_size, arguments.direction,
                                   arguments.limit, arguments.skip_bad, test_one_batch=arguments.test_one_batch,
                                   num_examples_print=arguments.num_examples_print)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id_key', default='tower_instruct', type=str,
                        choices=['tower_instruct', 'llama_chat', 'pseudo_quant'])
    parser.add_argument('--lang_pairs', type=str, nargs='+',
                        default=["en-fr_FR", "en-de_DE", "en-it_IT", "en-hi_IN", "en-sw_KE", "en-th_TH"])
    parser.add_argument('--precision', default='fp16',
                        type=str, help='Precision for which scores are being calculated if not Pseudo quantization',
                        choices=['fp16', 'bf16', 'int8', 'int4'])
    parser.add_argument('--pseudo_quant_precision', type=int, nargs='+',
                        default=list(range(1, 17)), help='Pseudo Quant model for which translation has to be run')
    parser.add_argument('--batch_size', default=16,
                        type=int, help='BatchSize to process')
    parser.add_argument('--direction', default='en2xx',
                        type=str, help='Translate English→Other (en2xx) or Other→English (xx2en). Default: en2xx. '
                                       'Only applicable for Tower model. For Llama default en2xx.',
                        choices=['en2xx', 'xx2en'])
    parser.add_argument('--limit', default=0,
                        type=int, help='If > 0, only translate the first N usable segments after filtering')
    parser.add_argument('--skip_bad', action='store_true',
                        help='If set, drop rows where is_bad_source == True (recommended)')
    parser.add_argument('--test_one_batch', action='store_false',
                        help='If set, only one batch is tested for translation')
    parser.add_argument('--num_examples_print', default=5, type=int,
                        help='Number of examples to print for a quick spot-check')
    parser.add_argument('--result_base_dir', default='../results/', type=str)
    args = parser.parse_args()
    main(args)
