from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import evaluate

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

# load dataset for specified lang pair
lang_pair = "en-fr_FR"
target_lang_code = lang_pair.split("-")[-1]
target_lang = LANGUAGE_BY_CODE[target_lang_code]
dataset = load_dataset("google/wmt24pp", lang_pair)['train'] # only 'train' dataset available

# --- MODEL ---
# load LLaMA2-7b
model_name = "meta-llama/Llama-2-7b-hf"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
# on Llama < 3.1 the default padding token in HF Tokenizer is None -> add it manually
# tokenizer.add_special_tokens({"pad_token": "<PAD>",})
# model.resize_token_embeddings(model.config.vocab_size + 1)

model.eval()

def translate(src_text, target_lang):  
    prompt = f"Translate this from English to {target_lang}: {src_text}"
    print(f"\nPrompt: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100, 
            do_sample=True,
            num_beams=5,
            repetition_penalty=1.5,
            length_penalty=1.5,
            early_stopping=True
        )
    
    decoded = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
    # decoded_translation = decoded_full[0].split(f"{target_lang}: ")[-1].strip()
    return decoded[0]


# --- GENERATE ---
# prep source & reference
sources = [ex for ex in dataset['source'][1:]]
targets = [ex for ex in dataset['target'][1:]]

# generate & store predictions 
predictions = []
n_examples = 5
for i, src in enumerate(sources[:n_examples]):
    translation = translate(src, target_lang)
    predictions.append(translation)
    # print(f'\nSource:\n\t{src}')
    print(f'\nTranslation:\n\t{translation}')
    print(f'\nTarget:\n\t{targets[i]}')
    print('---')

chrf = evaluate.load("chrf")
score = chrf.compute(predictions=predictions, references=targets[:n_examples])
print(f"chrF score: {score}")