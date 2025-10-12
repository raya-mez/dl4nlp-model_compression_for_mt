import re
import torch

# Language Data
LANG_PAIRS = ["en-fr_FR", "en-de_DE", "en-it_IT", "en-hi_IN", "en-sw_KE", "en-th_TH"]
LANG_CODES = [lp.split("-")[-1] for lp in LANG_PAIRS]

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

RESOURCE_MAP = {
    "fr_FR": "high",
    "de_DE": "high",
    "it_IT": "mid",
    "hi_IN": "mid",
    "sw_KE": "low",
    "th_TH": "low"
}

# Model Loading
model_id_map = {
    'tower_instruct': 'Unbabel/TowerInstruct-7B-v0.1',
    'llama_chat': 'meta-llama/Llama-2-7b-chat-hf'
}

# Generation Constants
MAX_NEW_TOKENS = 256

ASSISTANT_TAG_RE = re.compile(r"\[\/INST\]", re.IGNORECASE)
USER_BLOCK_RE = re.compile(r"<s>\s*\[INST\].*?\[\/INST\]", re.IGNORECASE | re.DOTALL)

# Global Parameters
max_allowed_torch_dtype = torch.float16
if max_allowed_torch_dtype == torch.float32:
    allowed_pseudo_max_bit = 32
else:
    allowed_pseudo_max_bit = 16
