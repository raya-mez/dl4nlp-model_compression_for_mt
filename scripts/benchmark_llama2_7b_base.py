import json
import os
import torch
import logging
import argparse
from datasets import load_dataset
from transformers import pipeline
from comet import download_model, load_from_checkpoint
from sacrebleu.metrics import CHRF

# --- SCRIPT CONFIG ---
# setup argument parser
parser = argparse.ArgumentParser(description="Benchmark Llama2-7B")
parser.add_argument('--out_dir', type=str, default="results_llama2_7b_base")
parser.add_argument('--lang_pairs', type=str, nargs='+', default=["en-fr_FR", "en-de_DE", "en-it_IT", "en-hi_IN", "en-sw_KE", "en-th_TH"])
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_new_tokens', type=int, default=256)

args = parser.parse_args()

LANG_PAIRS = args.lang_pairs
BATCH_SIZE = args.batch_size
MAX_NEW_TOKENS = args.max_new_tokens
OUT_DIR = args.out_dir

logging.basicConfig(level='INFO', format='%(asctime)s %(levelname)s %(message)s')
logging.info(f"Benchmarking LLaMA2-7b for lang pairs: {LANG_PAIRS}")
os.makedirs(OUT_DIR, exist_ok=True)
logging.info(f"Output directory: {OUT_DIR}")

# taken from the the WMT24++ dataset's HF page
LANGUAGE_BY_CODE = {
    "ar_EG": "Arabic","ar_SA": "Arabic","bg_BG": "Bulgarian","bn_BD": "Bengali","bn_IN": "Bengali",
    "ca_ES": "Catalan","cs_CZ": "Czech","da_DK": "Danish","de_DE": "German","el_GR": "Greek",
    "es_MX": "Spanish","et_EE": "Estonian","fa_IR": "Farsi","fi_FI": "Finnish","fil_PH": "Filipino",
    "fr_CA": "French","fr_FR": "French","gu_IN": "Gujarati","he_IL": "Hebrew","hi_IN": "Hindi",
    "hr_HR": "Croatian","hu_HU": "Hungarian","id_ID": "Indonesian","is_IS": "Icelandic","it_IT": "Italian",
    "ja_JP": "Japanese","kn_IN": "Kannada","ko_KR": "Korean","lt_LT": "Lithuanian","lv_LV": "Latvian",
    "ml_IN": "Malayalam","mr_IN": "Marathi","nl_NL": "Dutch","no_NO": "Norwegian","pa_IN": "Punjabi",
    "pl_PL": "Polish","pt_BR": "Portuguese","pt_PT": "Portuguese","ro_RO": "Romanian","ru_RU": "Russian",
    "sk_SK": "Slovak","sl_SI": "Slovenian","sr_RS": "Serbian","sv_SE": "Swedish","sw_KE": "Swahili",
    "sw_TZ": "Swahili","ta_IN": "Tamil","te_IN": "Telugu","th_TH": "Thai","tr_TR": "Turkish",
    "uk_UA": "Ukrainian","ur_PK": "Urdu","vi_VN": "Vietnamese","zh_CN": "Mandarin","zh_TW": "Mandarin",
    "zu_ZA": "Zulu"
} 

# --- LANGUAGE MODEL ---
pipe = pipeline(
    "text-generation",
    model="meta-llama/Llama-2-7b-hf",
    dtype=torch.bfloat16,
    device_map="auto"
)

tokenizer = pipe.tokenizer
# ensure tokenizer has a pad token for batching
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
# decoder-only models require left padding
tokenizer.padding_side = 'left'

# --- EVAL MODELS --- 
comet_load = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_load)
chrf = CHRF(word_order=2)


# --- TRANSLATION ---
def translate_dataset(lang_pair):
    # extract target language from language pair code
    tgt_lang_code = lang_pair.split("-")[-1]
    tgt_lang_name = LANGUAGE_BY_CODE.get(tgt_lang_code)

    logging.info(f"=== Translating English -> {tgt_lang_name} ({tgt_lang_code}) ===")

    # load dataset
    dataset = load_dataset("google/wmt24pp", lang_pair)["train"]

    # init variables to store generated translations, sources, and references
    translations = []
    srcs_refs_dict = []
    all_segment_ids = []
    
    for i in range(0, len(dataset), BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, len(dataset))
        logging.info(f"\nProcessing batch {i}:{batch_end} ({batch_end - i} items)")
        batch = dataset.select(range(i, batch_end))
        batch_prompts = []
        segment_ids = []
        batch_srcs_refs = []
        for entry in batch:
            src = entry["source"]
            ref = entry["target"]
            id = entry["segment_id"]
            # NOTE: plain prompt, no chat template
            # prompt based on Schmidtová et al. (2025)'s Prompt 3 (achieving highest COMET scores when non-perturbed)
            prompt = f"Translate this from English to {tgt_lang_name}:\n English: {src}\n {tgt_lang_name}:" 
            batch_prompts.append(prompt)
            srcs_refs_dict.append({"src": src, "ref": ref})
            batch_srcs_refs.append({"src": src, "ref": ref})
            segment_ids.append(id)
    
        # generate with EOS/PAD ids and extract continuation per item
        outputs = pipe(
            batch_prompts, 
            max_new_tokens=MAX_NEW_TOKENS, 
            do_sample=False,
            return_full_text=False,  # return only newly generated text
            batch_size=BATCH_SIZE
        )        
        # collect translations
        for out in outputs:
            hyp = out[0]["generated_text"].strip()
            translations.append(hyp)

        # log a few examples from this batch
        examples_to_log = min(3, len(outputs))
        for j in range(examples_to_log):
            src_example = batch_srcs_refs[j]["src"]
            ref_example = batch_srcs_refs[j]["ref"]
            hyp_example = outputs[j][0]["generated_text"].strip()
            logging.info(
                "\n--- Sample %d ---\nSRC: %s \nREF: %s \nHYP: %s",
                j + 1,
                src_example,
                ref_example,
                hyp_example,
            )
        
        all_segment_ids.extend(segment_ids)
        logging.info(f"Total translations so far: {len(translations)}")

    # save translations
    trans_out_file = os.path.join(OUT_DIR, f"predictions_{lang_pair}.txt")
    with open(trans_out_file, "w", encoding="utf-8") as pred_file:
        for id, pred in zip(all_segment_ids, translations):
            pred_file.write(f"{id}\t{pred}\n")
    logging.info(f"Saved translations to {trans_out_file}")

    # compute eval metrics
    refs = [entry["ref"] for entry in srcs_refs_dict]
    comet_input_data = [{"src": entry["src"], "mt": mt, "ref": entry["ref"]} for entry, mt in zip(srcs_refs_dict, translations)]

    logging.info("Scoring with COMET and CHRF++ ...")
    comet_output = comet_model.predict(comet_input_data, batch_size=32, gpus=1)
    chrf_score = chrf.corpus_score(translations, [refs])

    # save scores
    scores_out_file = os.path.join(OUT_DIR, f"scores_{tgt_lang_code}.json")
    with open(scores_out_file, "w", encoding="utf-8") as f:
        json.dump({
            "language": tgt_lang_code,
            "COMET": comet_output.system_score,
            "CHRF++": chrf_score.score
            }, f, ensure_ascii=False, indent=2)
        
    logging.info(f"Saved scores to {scores_out_file}")
    logging.info(f"Completed {tgt_lang_name}: COMET={comet_output.system_score:.4f}, CHRF++={chrf_score.score:.4f}")


for lp in LANG_PAIRS:
    translate_dataset(lp)
