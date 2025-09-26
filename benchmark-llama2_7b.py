"""This script benchmarks the Llama2 7B model's translation performance on a subset of the WMT24++ multilingual dataset."""

import json
import torch
import logging
import argparse
import evaluate
from datasets import load_dataset
from comet import download_model, load_from_checkpoint
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- SCRIPT CONFIG ---
# setup argument parser
parser = argparse.ArgumentParser(description="Benchmark Llama2 7B")
parser.add_argument('--scores_path', type=str, default='llama2_7b_scores.json')
parser.add_argument('--max_tokens', type=int, default=350, help='Maximum number of tokens to generate')
parser.add_argument('--n_examples', type=int, default=None, help='Number of examples from the dataset to use. Defaults to the size of the dataset')
# subset of languages chosen to test on (default: 2 of high-, medium-, and low-resource, in order)
parser.add_argument('--lang_codes', type=str, nargs='+', default=["fr_FR", "de_DE", "it_IT", "hi_IN", "sw_KE", "th_TH"], help='Languages to benchmark translation for. Indicate by their codes.')

args = parser.parse_args()

max_tokens = args.max_tokens
scores_path = args.scores_path
n_examples = args.n_examples
lang_codes = args.lang_codes

logging.basicConfig(level='INFO')
logging.info(f"Running LLaMA2-7b benchmark with max_tokens={max_tokens}, n_examples={n_examples}, lang_codes={lang_codes}")


# --- DATA CONFIG ---
# dataset constants
LANGUAGE_BY_CODE = {
    "ar_EG": "Arabic",
    "ar_SA": "Arabic",
    "bg_BG": "Bulgarian",
    "bn_BD": "Bengali",
    "bn_IN": "Bengali",
    "ca_ES": "Catalan",
    "cs_CZ": "Czech",
    "da_DK": "Danish",
    "de_DE": "German",
    "el_GR": "Greek",
    "es_MX": "Spanish",
    "et_EE": "Estonian",
    "fa_IR": "Farsi",
    "fi_FI": "Finnish",
    "fil_PH": "Filipino",
    "fr_CA": "French",
    "fr_FR": "French",
    "gu_IN": "Gujarati",
    "he_IL": "Hebrew",
    "hi_IN": "Hindi",
    "hr_HR": "Croatian",
    "hu_HU": "Hungarian",
    "id_ID": "Indonesian",
    "is_IS": "Icelandic",
    "it_IT": "Italian",
    "ja_JP": "Japanese",
    "kn_IN": "Kannada",
    "ko_KR": "Korean",
    "lt_LT": "Lithuanian",
    "lv_LV": "Latvian",
    "ml_IN": "Malayalam",
    "mr_IN": "Marathi",
    "nl_NL": "Dutch",
    "no_NO": "Norwegian",
    "pa_IN": "Punjabi",
    "pl_PL": "Polish",
    "pt_BR": "Portuguese",
    "pt_PT": "Portuguese",
    "ro_RO": "Romanian",
    "ru_RU": "Russian",
    "sk_SK": "Slovak",
    "sl_SI": "Slovenian",
    "sr_RS": "Serbian",
    "sv_SE": "Swedish",
    "sw_KE": "Swahili",
    "sw_TZ": "Swahili",
    "ta_IN": "Tamil",
    "te_IN": "Telugu",
    "th_TH": "Thai",
    "tr_TR": "Turkish",
    "uk_UA": "Ukrainian",
    "ur_PK": "Urdu",
    "vi_VN": "Vietnamese",
    "zh_CN": "Mandarin",
    "zh_TW": "Mandarin",
    "zu_ZA": "Zulu",
}
LANGUAGE_PAIRS = (
    "en-ar_EG", "en-ar_SA", "en-bg_BG", "en-bn_IN", "en-ca_ES", "en-cs_CZ", "en-da_DK", "en-de_DE",
    "en-el_GR", "en-es_MX", "en-et_EE", "en-fa_IR", "en-fi_FI", "en-fil_PH", "en-fr_CA", "en-fr_FR",
    "en-gu_IN", "en-he_IL", "en-hi_IN", "en-hr_HR", "en-hu_HU", "en-id_ID", "en-is_IS", "en-it_IT",
    "en-ja_JP", "en-kn_IN", "en-ko_KR", "en-lt_LT", "en-lv_LV", "en-ml_IN", "en-mr_IN", "en-nl_NL",
    "en-no_NO", "en-pa_IN", "en-pl_PL", "en-pt_BR", "en-pt_PT", "en-ro_RO", "en-ru_RU", "en-sk_SK",
    "en-sl_SI", "en-sr_RS", "en-sv_SE", "en-sw_KE", "en-sw_TZ", "en-ta_IN", "en-te_IN", "en-th_TH",
    "en-tr_TR", "en-uk_UA", "en-ur_PK", "en-vi_VN", "en-zh_CN", "en-zh_TW", "en-zu_ZA",
)

# subset of language pairs from the dataset to use for testing given specified lang_codes
lang_pairs = [f"en-{lc}" for lc in lang_codes]


# --- MODEL ---
# load LLaMA2-7b & tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()
# ---

# --- TRANSLATE() ---
# function to generate and decode translations
def translate(src_text, src_lang, tgt_lang):  
    # prompt = f"{src_lang}: {src_text}\n{tgt_lang}: "
    prompt = f"Translate this from {src_lang} to {tgt_lang}:\n {src_lang}: {src_text}\n {tgt_lang}:" # based on Schmidtová et al. (2025)'s Prompt 3 (achieving highest COMET scores when non-perturbed)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens, 
            num_beams=5, 
            early_stopping=True
        )
    
    # decode only the generated tokens after the prompt tokens
    generated_tokens = outputs[:, inputs["input_ids"].shape[-1] :]
    decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    
    return decoded
# ---


# --- LOAD COMET ---
# COMET model for translation evaluation
comet_model_name = "Unbabel/wmt22-comet-da" 
model_path = download_model(comet_model_name)
comet_model = load_from_checkpoint(model_path)
# ---


# dictionary that will store evaluation scores
scores = {}

# --- INFERENCE LOOP ---
# iterate over all lang pairs, generate, and evaluate translations
for l_code, l_pair in zip(lang_codes, lang_pairs): 
    logging.info(f"Running benchmark for language pair: {l_pair}...")

    # --- PREP DATA ---
    # load dataset for specified lang pair
    dataset = load_dataset("google/wmt24pp", l_pair)['train'] # only 'train' dataset available (instances are called 'test-xxx")
    target_lang = LANGUAGE_BY_CODE[l_code]

    # prep source & reference examples, skipping sources where 'is_bad_source' is True
    sources = []
    targets = []
    predictions = []
    prediction_ids = []

    # --- GENERATE ---
    # generate & store predictions   
    for i, ex in enumerate(dataset):
        if n_examples is not None and len(predictions) == n_examples:
            break
        if i % 100 == 0:
            logging.info(f"\nTranslation iteration {i}/{n_examples if n_examples else len(dataset)}...")
        if not ex['is_bad_source'] == True:
            translation = translate(ex['source'], "English", target_lang)
            predictions.append(translation) 
            prediction_ids.append(ex['segment_id'])
            sources.append(ex['source'])
            targets.append(ex['target'])
    
    # --- SAVING PREDICTIONS ---
    # write predictions and corresponding segment_id's to a file
    pred_out_path = f"predictions_{l_pair}.txt"
    with open(pred_out_path, "w", encoding="utf-8") as pred_file:
        for id, pred in zip(prediction_ids, predictions):
            pred_file.write(f"{id}\t{pred}\n")
    logging.info(f"Saved predictions to {pred_out_path}")

    # --- EVAL ---
    # compute chrF++ score
    chrf = evaluate.load("chrf")
    chrf_score = chrf.compute(
        predictions=predictions, 
        references=targets,
        word_order=2  # for chrF++ score
        )
    logging.info(f"chrF++ score: {chrf_score}")
    
    # compute COMET score
    comet_data = [
        {"src": src, "mt": pred, "ref": ref}
        for src, pred, ref in zip(sources, predictions, targets)
    ]
    comet_score = comet_model.predict(comet_data, gpus=1).system_score  # overall score (instead of scores for individual examples)
    logging.info(f"COMET score: {comet_score}")

    # store chrF & COMET scores for the current lang
    scores[l_pair] = {
        "num_predictions": len(predictions),
        "chrf_score": chrf_score['score'],
        "comet_score": comet_score
    }
# ---

# --- SAVING EVAL SCORES ---
# write scores to json file
with open(scores_path, "w", encoding="utf-8") as f:
    json.dump(scores, f)

logging.info(f"Saved chrF++ and COMET scores in {scores_path}")