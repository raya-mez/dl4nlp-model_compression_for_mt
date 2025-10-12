import os
import re
import glob
import torch
from sacrebleu.metrics import CHRF, BLEU
from comet import download_model, load_from_checkpoint

from constants import USER_BLOCK_RE, ASSISTANT_TAG_RE, allowed_pseudo_max_bit


def load_comet_model():
    comet_load = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_load)
    return comet_model


def load_chrf_model():
    chrf_model = CHRF(word_order=2)
    return chrf_model


def load_bleu_model(effective_order=False):
    if effective_order:
        bleu_model = BLEU(effective_order=effective_order)
    else:
        bleu_model = BLEU()
    return bleu_model


def find_translation_files(folder_path):
    """Find all JSONL files that start with 'translations_' in the given folder"""
    pattern = os.path.join(folder_path, "translations_*.jsonl")
    translation_files = glob.glob(pattern)
    return translation_files


def extract_language_code(filename):
    """Extract language code from filename like translations_fr_FR_en2xx_fp16.jsonl"""
    base_name = os.path.basename(filename)
    lang_code = base_name.replace("translations_", "").replace(".jsonl", "")
    return lang_code


def clean_translation(text, target_lang_name):
    """Return only the translation text (no chat tags, no prompt)."""
    if not text:
        return text

    t = text.strip()

    # (handles cases where the model reprints the prompt before answering).
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


class KBitConfig:
    def __init__(self, k, group_size, f_dtype):
        assert k > 0, "k must be greater than 0"
        assert k <= allowed_pseudo_max_bit, f"k must be lesser than equal to {allowed_pseudo_max_bit}"
        self.k = k
        self.group_size = group_size
        self.f_dtype = f_dtype


def smallest_int_dtype_for_k(k):
    if k <= 8:
        return torch.int8
    elif 8 < k <= 16:
        return torch.int16
    else:
        return torch.int32


def kbit_range(k):
    if k == 1:
        return -1, 1
    else:
        qmax = (1 << (k - 1)) - 1
        qmin = - (1 << (k - 1))
        return qmin, qmax
