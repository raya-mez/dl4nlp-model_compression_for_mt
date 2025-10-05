import os
import json
import pandas as pd
from sacrebleu.metrics import CHRF, BLEU
from comet import download_model, load_from_checkpoint
import numpy as np
import glob
import logging

# Reduce verbosity of PyTorch Lightning and other logs
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("sacrebleu").setLevel(logging.WARNING)

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

# Configuration: Define paths for each precision model
PRECISION_PATHS = {
    "fp16": "/content/fp16 results",      # Update these paths as needed
    "int8": "/content/int8 results",      # Update these paths as needed  
    "int4": "/content/int4 results"       # Update these paths as needed
}

# Load evaluation models
print("Loading COMET model...")
comet_load = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(comet_load)
chrf = CHRF(word_order=2)
bleu = BLEU()

def compute_sentence_level_scores(mt_data):
    # Prepare batch data for all metrics
    hypotheses = [item["hyp"] for item in mt_data]
    references = [item["ref"] for item in mt_data]
    comet_inputs = []
    
    for item in mt_data:
        src = item["src"]
        hyp = item["hyp"]
        ref = item["ref"]
        comet_inputs.append({"src": src, "mt": hyp, "ref": ref})

    # Compute COMET scores in batch for all sentences
    print(f"  Computing COMET scores for {len(comet_inputs)} sentences...")
    comet_output = comet_model.predict(comet_inputs, batch_size=16, gpus=1)
    comet_scores = comet_output["scores"]

    # Compute chrF++ scores
    print(f"  Computing chrF++ scores for {len(hypotheses)} sentences...")
    chrf_scores = []
    for hyp, ref in zip(hypotheses, references):
        chrf_score = chrf.sentence_score(hyp, [ref]).score
        chrf_scores.append(chrf_score)

    # Compute sentence-level BLEU scores
    print(f"  Computing BLEU scores for {len(hypotheses)} sentences...")
    bleu_scores = []
    # For sentence-level BLEU, we need to handle the effective_order warning differently
    # We'll create a temporary BLEU scorer with effective_order enabled
    temp_bleu = BLEU(effective_order=True)
    for hyp, ref in zip(hypotheses, references):
        try:
            bleu_score = temp_bleu.sentence_score(hyp, [ref]).score
            bleu_scores.append(bleu_score)
        except Exception as e:
            # Fallback to regular BLEU if effective_order causes issues
            bleu_score = bleu.sentence_score(hyp, [ref]).score
            bleu_scores.append(bleu_score)

    # Store scores and segment ids
    per_sent_scores = {
        "segment_id": [item["segment_id"] for item in mt_data],
        "COMET": comet_scores,
        "chrF++": chrf_scores,
        "BLEU": bleu_scores
    }
    return per_sent_scores

def compute_overall_scores(mt_data, per_sent_scores):
    """Compute overall corpus-level scores including BLEU"""
    # Extract all hypotheses and references
    hypotheses = [item["hyp"] for item in mt_data]
    references = [item["ref"] for item in mt_data]
    
    # Compute corpus-level BLEU score
    bleu_score = bleu.corpus_score(hypotheses, [references])
    
    # Compute averages for all metrics
    avg_comet = np.mean(per_sent_scores["COMET"])
    avg_chrf = np.mean(per_sent_scores["chrF++"])
    avg_bleu = np.mean(per_sent_scores["BLEU"])
    
    return {
        "BLEU_corpus": bleu_score.score,
        "BLEU_sent_avg": avg_bleu,
        "COMET": avg_comet,
        "chrF++": avg_chrf,
        "BLEU_details": str(bleu_score)
    }

def find_translation_files(folder_path):
    """Find all JSONL files that start with 'translations_' in the given folder"""
    pattern = os.path.join(folder_path, "translations_*.jsonl")
    translation_files = glob.glob(pattern)
    return translation_files

def extract_language_code(filename):
    """Extract language code from filename like translations_fr_FR.jsonl"""
    base_name = os.path.basename(filename)
    lang_code = base_name.replace("translations_", "").replace(".jsonl", "")
    return lang_code

def main():
    # Store overall results for all languages and precisions
    overall_results = []
    
    for precision, precision_path in PRECISION_PATHS.items():
        print(f"\n=== START {precision} ===")
        print(f"Path: {precision_path}")
        
        if not os.path.exists(precision_path):
            print(f"Warning: Path {precision_path} does not exist. Skipping {precision}.")
            continue
        
        translation_files = find_translation_files(precision_path)
        
        if not translation_files:
            print(f"Warning: No translation files found in {precision_path}")
            continue
            
        print(f"Found {len(translation_files)} translation file(s)")
        
        for file_path in translation_files:
            lang_code = extract_language_code(file_path)
            
            if lang_code not in LANG_CODES:
                print(f"Warning: Language code '{lang_code}' from file {file_path} not in expected list. Processing anyway.")
            
            print(f"[{lang_code} {precision}] Processing {os.path.basename(file_path)}...")
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = [json.loads(line) for line in f]
                
                print(f"  Found {len(data)} sentences")
                
                # Compute all scores
                per_sent_scores = compute_sentence_level_scores(data)
                overall_scores = compute_overall_scores(data, per_sent_scores)
                
                # Add metadata
                overall_scores.update({
                    "precision": precision,
                    "language": lang_code,
                    "language_name": LANGUAGE_BY_CODE.get(lang_code, "Unknown"),
                    "resource_level": RESOURCE_MAP.get(lang_code, "unknown"),
                    "file_path": file_path
                })
                
                overall_results.append(overall_scores)
                
                # Save sentence-level scores
                per_sent_file = os.path.join(precision_path, f"per_sentence_scores_{lang_code}.csv")
                pd.DataFrame(per_sent_scores).to_csv(per_sent_file, index=False)
                print(f"  Saved per-sentence scores to {os.path.basename(per_sent_file)}")
                
                # Print results
                print(f"  Scores - BLEU (corpus): {overall_scores['BLEU_corpus']:.2f}, "
                      f"BLEU (sent avg): {overall_scores['BLEU_sent_avg']:.2f}, "
                      f"COMET: {overall_scores['COMET']:.4f}, "
                      f"chrF++: {overall_scores['chrF++']:.2f}")
                      
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save and display overall results
    if overall_results:
        overall_df = pd.DataFrame(overall_results)
        
        column_order = ['precision', 'language', 'language_name', 'resource_level', 
                       'BLEU_corpus', 'BLEU_sent_avg', 'COMET', 'chrF++', 'BLEU_details', 'file_path']
        column_order = [col for col in column_order if col in overall_df.columns]
        overall_df = overall_df[column_order]
        
        overall_file = "overall_scores_summary.csv"
        overall_df.to_csv(overall_file, index=False)
        print(f"\n=== OVERALL SUMMARY ===")
        print(f"Saved overall scores summary to {overall_file}")
        
        # Display summary tables
        print("\n" + "="*100)
        print("OVERALL SCORES SUMMARY")
        print("="*100)
        summary_df = overall_df[['precision', 'language', 'BLEU_corpus', 'BLEU_sent_avg', 'COMET', 'chrF++']].copy()
        summary_df = summary_df.round({'BLEU_corpus': 2, 'BLEU_sent_avg': 2, 'COMET': 4, 'chrF++': 2})
        print(summary_df.to_string(index=False))
        
    else:
        print("No results were processed. Please check your file paths.")

if __name__ == "__main__":
    main()