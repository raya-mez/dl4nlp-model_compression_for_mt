import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import json
import logging

from utils import load_comet_model, load_chrf_model, load_bleu_model, find_translation_files, extract_language_code
from constants import LANGUAGE_BY_CODE, RESOURCE_MAP

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("sacrebleu").setLevel(logging.WARNING)


comet_model = load_comet_model()
chrf_model = load_chrf_model()
bleu_model = load_bleu_model()


def compute_sentence_level_scores(segment_ids, sources, hypotheses, references):
    comet_inputs = []
    for index in range(len(sources)):
        src = sources[index]
        hyp = hypotheses[index]
        ref = references[index]
        comet_inputs.append({"src": src, "mt": hyp, "ref": ref})

    # Compute COMET scores in batch for all sentences
    print(f"  Computing COMET scores for {len(comet_inputs)} sentences...")
    comet_output = comet_model.predict(comet_inputs, batch_size=16, gpus=1)
    comet_scores = comet_output["scores"]

    # Compute chrF++ scores
    print(f"  Computing chrF++ scores for {len(hypotheses)} sentences...")
    chrf_scores = []
    for hyp, ref in zip(hypotheses, references):
        chrf_score = chrf_model.sentence_score(hyp, [ref]).score
        chrf_scores.append(chrf_score)

    # Compute sentence-level BLEU scores
    print(f"  Computing BLEU scores for {len(hypotheses)} sentences...")
    bleu_scores = []
    # For sentence-level BLEU, we need to handle the effective_order warning differently
    # We'll create a temporary BLEU scorer with effective_order enabled
    temp_bleu = load_bleu_model(effective_order=True)
    for hyp, ref in zip(hypotheses, references):
        try:
            bleu_score = temp_bleu.sentence_score(hyp, [ref]).score
            bleu_scores.append(bleu_score)
        except Exception as e:
            # Fallback to regular BLEU if effective_order causes issues
            bleu_score = bleu_model.sentence_score(hyp, [ref]).score
            bleu_scores.append(bleu_score)

    # Store scores and segment ids
    per_sent_scores = {
        "segment_id": segment_ids,
        "COMET": comet_scores,
        "chrF++": chrf_scores,
        "BLEU": bleu_scores
    }
    return per_sent_scores


def compute_corpus_level_scores(hypotheses, references, sources=None, include_comet=False):
    bleu_score = bleu_model.corpus_score(hypotheses, [references])
    chrf_score = chrf_model.corpus_score(hypotheses, [references])
    if include_comet:
        comet_inputs = []
        for index in range(len(sources)):
            src = sources[index]
            hyp = hypotheses[index]
            ref = references[index]
            comet_inputs.append({"src": src, "mt": hyp, "ref": ref})

        comet_output = comet_model.predict(comet_inputs, batch_size=16, gpus=1)

        return {'COMET_Corpus': comet_output, 'BLEU_Corpus': bleu_score, 'CHRF_Corpus': chrf_score}
    else:
        return {'BLEU_Corpus': bleu_score, 'CHRF_Corpus': chrf_score}


def compute_overall_scores(per_sent_scores, corpus_scores):
    """Compute overall corpus-level scores including BLEU"""
    
    # Compute sent level averages for all metrics
    avg_comet = np.mean(per_sent_scores["COMET"])
    avg_chrf = np.mean(per_sent_scores["chrF++"])
    avg_bleu = np.mean(per_sent_scores["BLEU"])
    
    return {
        "BLEU_corpus": corpus_scores['BLEU_Corpus'].score,
        "BLEU_sent_avg": avg_bleu,
        "chrF++_corpus": corpus_scores['CHRF_Corpus'].score,
        "chrF++_sent_avg": avg_chrf,
        "COMET": avg_comet,
        "BLEU_details": str(corpus_scores['BLEU_Corpus'])
    }


def create_summary_df(overall_results, precision_result_path):
    overall_df = pd.DataFrame(overall_results)

    column_order = ['precision', 'language', 'language_name', 'resource_level',
                    'COMET', 'BLEU_corpus', 'BLEU_sent_avg', 'chrF++_corpus', 'chrF++_sent_avg', 'BLEU_details',
                    'file_path']
    column_order = [col for col in column_order if col in overall_df.columns]
    overall_df = overall_df[column_order]

    overall_file_save_path = os.path.join(precision_result_path, "overall_scores_summary.csv")
    overall_df.to_csv(overall_file_save_path, index=False)

    summary_df = overall_df[['precision', 'language', 'COMET', 'BLEU_corpus', 'BLEU_sent_avg', 'chrF++_corpus',
                             'chrF++_sent_avg']].copy()
    summary_df['COMET'] = summary_df['COMET'].apply(lambda x: np.round(x, 4))
    summary_df['chrF++_corpus'] = summary_df['chrF++_corpus'].apply(lambda x: np.round(x, 2))
    summary_df['chrF++_sent_avg'] = summary_df['chrF++_sent_avg'].apply(lambda x: np.round(x, 2))
    summary_df['BLEU_corpus'] = summary_df['BLEU_corpus'].apply(lambda x: np.round(x, 2))
    summary_df['BLEU_sent_avg'] = summary_df['BLEU_sent_avg'].apply(lambda x: np.round(x, 2))

    save_path = os.path.join(precision_result_path, f"score_summary_df.csv")
    summary_df.to_csv(save_path, index=False)


def main(arguments):
    translation_files = find_translation_files(arguments.precision_result_path)
    if not translation_files:
        print(f"Warning: No translation files found in {arguments.precision_result_path}")
        return

    print(f"Found {len(translation_files)} translation file(s)")

    path = Path(arguments.precision_result_path)
    parts = path.parts

    precision = parts[-1]

    if parts[-2] == 'llama':
        direction = 'en2xx'
    else:
        direction = parts[-2]

    overall_results = []
    for file_path in translation_files:
        lang_code = extract_language_code(file_path)
        base_save_suffix = f"{lang_code}_{direction}_{precision}"
        print(f"[{lang_code} - {arguments.precision}] Processing {os.path.basename(file_path)}...")
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = [json.loads(line) for line in file]

            print(f"Found {len(data)} sentences")

            # Compute all scores
            segment_ids = [item["segment_id"] for item in data]
            sources = [item["src"] for item in data]
            hypotheses = [item["hyp"] for item in data]
            references = [item["ref"] for item in data]
            per_sent_scores = compute_sentence_level_scores(segment_ids, sources, hypotheses, references)
            corpus_scores = compute_corpus_level_scores(hypotheses, references)
            overall_scores = compute_overall_scores(per_sent_scores, corpus_scores)

            # Add metadata
            overall_scores.update({
                "precision": arguments.precision,
                "language": lang_code,
                "language_name": LANGUAGE_BY_CODE.get(lang_code, "Unknown"),
                "resource_level": RESOURCE_MAP.get(lang_code, "unknown"),
                "file_path": file_path
            })

            overall_results.append(overall_scores)

            # Save sentence-level scores
            per_sent_file = os.path.join(arguments.precision_result_path,
                                         f"per_sentence_scores_{base_save_suffix}.csv")
            pd.DataFrame(per_sent_scores).to_csv(per_sent_file, index=False)
            print(f"Saved per-sentence scores to {os.path.basename(per_sent_file)}")

            # Print results
            print(f"Scores - BLEU (corpus): {overall_scores['BLEU_corpus']:.2f}, "
                  f"BLEU (sent avg): {overall_scores['BLEU_sent_avg']:.2f}, "
                  f"COMET: {overall_scores['COMET']:.4f}, "
                  f"chrF++: {overall_scores['chrF++']:.2f}")
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

        if arguments.create_summary_df == 'yes' and len(overall_results) != 0:
            create_summary_df(overall_results, arguments.precision_result_path)
        else:
            print("No results were processed. Either was Pseudo Quantization or please check your file paths.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--precision', default='fp16',
                        type=str, help='Precision for which scores are being calculated if not Pseudo quantization',
                        choices=['fp16', 'int8', 'int4'])
    parser.add_argument('--precision_result_path',
                        default='../results/tower/en_2_xx/fp16/',
                        type=str, help='Path to the result folder containing the translations')
    parser.add_argument('--create_summary_df', default='yes',
                        type=str, help='Whether Summary DF is required after score calculation or not')
    args = parser.parse_args()
    main(args)
