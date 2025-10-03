import os
import json
import pandas as pd
from sacrebleu.metrics import CHRF
from comet import download_model, load_from_checkpoint

# Constants
LANG_PAIRS = ["en-fr_FR", "en-de_DE", "en-it_IT", "en-hi_IN", "en-sw_KE", "en-th_TH"]
LANG_CODES = [lp.split("-")[-1] for lp in LANG_PAIRS]

LANGUAGE_BY_CODE = {
    "de_DE": "German", "fr_FR": "French", 
    "hi_IN": "Hindi", "it_IT": "Italian", 
    "sw_KE": "Swahili", "th_TH": "Thai"
} 

RESOURCE_MAP = {
    "fr_FR": "high", 
    "de_DE": "high", 
    "it_IT": "mid", 
    "hi_IN": "mid", 
    "sw_KE": "low", 
    "th_TH": "low"
}

PRECISIONS = ["fp16", "int8", "int4"]

# Load evaluation models
comet_load = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_load)
chrf = CHRF(word_order=2)


# Compute COMET and chrF++ scores for each item in the data and save the results to a csv file
def compute_sentence_level_scores(mt_data):
    comet_scores = []
    chrf_scores = []

    for item in mt_data:
        src = item["src"]
        hyp = item["hyp"]
        ref = item["ref"]

        # Compute COMET score
        comet_input = {"src": src, "mt": hyp, "ref": ref}
        comet_score = comet_model.predict([comet_input], batch_size=16, gpus=1)["scores"][0]
        comet_scores.append(comet_score)

        # Compute chrF++ score
        chrf_score = chrf.sentence_score(hyp, [ref]).score
        chrf_scores.append(chrf_score)

    # Store scores and segment ids
    per_sent_scores = {
        "segment_id": [item["segment_id"] for item in mt_data],
        "COMET": comet_scores,
        "chrF++": chrf_scores
    }
    return per_sent_scores

def main():
    os.chdir("../results")
    
    for precision in PRECISIONS:
        print(f"\n=== START {precision} ===")
        for lang in LANG_CODES:
            # Find file and load data
            file = os.path.join(f"llama_{precision}", f"translations_{lang}.jsonl")
            print(f"[START {lang} {precision}] Processing {file}...")
            with open(file, "r", encoding="utf-8") as f:
                data = [json.loads(line) for line in f]
            
            # Compute per-sentence COMET & chrF++ scores
            per_sent_scores = compute_sentence_level_scores(data)
            
            # Save scores to csv
            per_sent_file = os.path.join(f"llama_{precision}", f"per_sentence_scores_{lang}.csv")
            pd.DataFrame(per_sent_scores).to_csv(per_sent_file, index=False)
            print(f"[END {lang} {precision}] Saved per-sentence scores to {per_sent_file}") 


if __name__ == "__main__":
    main()