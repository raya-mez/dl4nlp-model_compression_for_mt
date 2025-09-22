import json
import torch
import logging
import argparse
import evaluate
from datasets import load_dataset
from comet import download_model, load_from_checkpoint
from transformers import AutoTokenizer, AutoModelForCausalLM


parser = argparse.ArgumentParser(description="Benchmark Llama2 7B")
parser.add_argument('--scores_path', type=str, default='llama2_7b_scores.json')
parser.add_argument('--max_tokens', type=int, default=350, help='Maximum number of tokens to generate')
parser.add_argument('--n_examples', type=int, default=-1, help='Number of examples from the dataset to consider')
parser.add_argument('--n_langs', type=int, default=-1, help='Number of language pairs from the dataset to consider')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for batched generation')
args = parser.parse_args()

max_tokens = args.max_tokens
scores_path = args.scores_path
n_examples = args.n_examples
n_langs = args.n_langs
batch_size = args.batch_size

logging.basicConfig(level=logging.INFO)
logging.info(f"Running LLaMA2-7b benchmark with max_tokens={max_tokens}, n_examples={n_examples}, n_langs={n_langs}, batch_size={batch_size}")

# --- DATASET ---
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

# load LLaMA2-7b & tokenizer
model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# ensure pad token exists for batched padding
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
model.eval()

# function to generate and decode translations
def translate(src_text, src_lang, tgt_lang):  
    # prompt = f"{src_lang}: {src_text}\n{tgt_lang}: "
    prompt = f"Translate this from {src_lang} to {tgt_lang}:\n {src_lang}: {src_text}\n {tgt_lang}:"
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

# batched translation
def translate_batch(src_texts, src_lang, tgt_lang):
    prompts = [
        f"Translate this from {src_lang} to {tgt_lang}:\n {src_lang}: {text}\n {tgt_lang}:"
        for text in src_texts
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            num_beams=5,
            early_stopping=True
        )
    # decode only the generated tokens after prompt tokens per sample
    input_lengths = (inputs["input_ids"] != tokenizer.pad_token_id).sum(dim=1)
    decoded = []
    for i in range(outputs.size(0)):
        gen_tokens = outputs[i, input_lengths[i]:]
        decoded.append(tokenizer.decode(gen_tokens, skip_special_tokens=True))
    return decoded

comet_model_name = "Unbabel/wmt22-comet-da" 
model_path = download_model(comet_model_name)
comet_model = load_from_checkpoint(model_path)

# dictionary that will store evaluation scores
scores = {}

# iterate over all lang pairs, generate, and evaluate translations
for lang_pair in LANGUAGE_PAIRS[:n_langs]: 
    logging.info(f"Running benchmark for language pair: {lang_pair}...")

    # load dataset for specified lang pair
    target_lang_code = lang_pair.split("-")[-1]
    target_lang = LANGUAGE_BY_CODE[target_lang_code]
    dataset = load_dataset("google/wmt24pp", lang_pair)['train'] # only 'train' dataset available (instances are called 'test-xxx")

    # prep source & reference instances, skipping sources where 'is_bad_source' is True
    sources = []
    targets = []
    for src, tgt, is_bad_source in zip(dataset['source'], dataset['target'], dataset['is_bad_source']):
        if not is_bad_source:
            sources.append(src)
            targets.append(tgt)

    # generate & store predictions (batched)
    predictions = []
    effective_n = len(sources[:n_examples]) if n_examples != -1 else len(sources)
    for start in range(0, effective_n, batch_size):
        end = min(start + batch_size, effective_n)
        batch_src = sources[start:end]
        batch_preds = translate_batch(batch_src, "English", target_lang)
        predictions.extend(batch_preds)
        # keep previous logging of first 5 examples
        for j, pred in enumerate(batch_preds):
            global_index = start + j
            if global_index < 5:
                logging.info(
                    f"Model output for example {global_index + 1}:\n{pred}\nSource: {sources[global_index]}\nTarget: {targets[global_index]}"
                )

    # compute chrF score
    chrf = evaluate.load("chrf")
    chrf_score = chrf.compute(
        predictions=predictions, 
        references=targets[:n_examples],
        word_order=2
        )
    logging.info(f"chrF++ score: {chrf_score}")
    
    # compute COMET score
    comet_data = [
        {"src": src, "mt": pred, "ref": ref}
        for src, pred, ref in zip(sources[:n_examples], predictions, targets[:n_examples])
    ]
    comet_score = comet_model.predict(comet_data).system_score
    logging.info(f"COMET score: {comet_score}")
    examples = len(sources[:n_examples])

    # store chrF & COMET scores for the current lang
    scores[lang_pair] = (examples, chrf_score, comet_score)

# write scores to json file
with open(scores_path, "w", encoding="utf-8") as f:
    json.dump(scores, f)

logging.info(f"Saved chrF++ & COMET scores for all lang pairs in {scores_path}")