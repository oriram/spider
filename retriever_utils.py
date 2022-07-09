import csv
import glob
import json
import gzip
import logging
import functools
from pathlib import Path

import wandb
from typing import List, Tuple, Dict, Iterator, Union

from dpr.data.qa_validation import calculate_matches

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

RECALL_FILE_NAME = "recall_at_k.csv"
RESULTS_FILE_NAME = "results.json"


def parse_qa_csv_file(location) -> Iterator[Tuple[str, List[str]]]:
    with open(location) as ifile:
        reader = csv.reader(ifile, delimiter="\t")
        for row in reader:
            question = row[0]
            answers = eval(row[1])
            yield question, answers


def parse_qa_json_file(path):
    with open(path, "r") as f:
        data = json.load(f)
    for d in data:
        question = d["question"]
        answers = d["answers"]
        if "entity" in d:
            yield question, answers, d["entity"]
        else:
            yield question, answers

def validate(
    dataset_name: str,
    passages: Dict[object, Tuple[str, str]],
    answers: List[List[str]],
    result_ctx_ids: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
    out_file: str,
    use_wandb: bool = True,
    output_recall_at_k: bool = False,
    log: bool = True
) -> Union[List[List[bool]], Tuple[object, List[float]]]:
    match_stats = calculate_matches(
        passages, answers, result_ctx_ids, workers_num, match_type, log=log
    )
    top_k_hits = match_stats.top_k_hits

    if log: logger.info("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(result_ctx_ids) for v in top_k_hits]
    if log: logger.info("Validation results: top k documents hits accuracy %s", top_k_hits)
    with open(out_file, "w") as f:
        for k, recall in enumerate(top_k_hits):
            f.write(f"{k+1},{recall}\n")
            if use_wandb:
                wandb.log({f"eval-{dataset_name}/k": k+1, f"eval-{dataset_name}/recall": recall})
    if log: logger.info(f"Saved recall@k info to {out_file}")
    return match_stats.questions_doc_hits if not output_recall_at_k else (match_stats.questions_doc_hits, top_k_hits)


def load_passages(ctx_file: str) -> Dict[object, Tuple[str, str]]:
    docs = {}
    logger.info("Reading data from: %s", ctx_file)
    if ctx_file.endswith(".gz"):
        with gzip.open(ctx_file, "rt") as tsvfile:
            reader = csv.reader(
                tsvfile,
                delimiter="\t",
            )
            # file format: doc_id, doc_text, title
            for row in reader:
                if row[0] != "id":
                    docs[row[0]] = (row[1], row[2])
    else:
        with open(ctx_file) as tsvfile:
            reader = csv.reader(
                tsvfile,
                delimiter="\t",
            )
            # file format: doc_id, doc_text, title
            for row in reader:
                if row[0] != "id":
                    docs[row[0]] = (row[1], row[2])
    return docs


def save_results(
    passages: Dict[object, Tuple[str, str]],
    questions: List[str],
    answers: List[List[str]],
    top_passages_and_scores: List[Tuple[List[object], List[float]]],
    per_question_hits: List[List[bool]],
    out_file: str,
    output_no_text: bool = False,
):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        hit_indices = [j+1 for j, is_hit in enumerate(hits) if is_hit]
        hit_min_rank = hit_indices[0] if len(hit_indices) > 0 else None
        ctxs_num = len(hits)

        d = {
                "question": q,
                "answers": q_answers,
                "hit_min_rank": hit_min_rank,
                "all_hits": hit_indices,
                "ctxs": [
                    {
                        "id": results_and_scores[0][c],
                        "rank": (c + 1),
                        "title": docs[c][1],
                        "text": docs[c][0] if not output_no_text else "",
                        "score": scores[c],
                        "has_answer": hits[c],
                    }
                    for c in range(ctxs_num)
                ],
            }
        merged_data.append(d)

    with open(out_file, "w") as writer:
        writer.write(json.dumps(merged_data, indent=4) + "\n")
    logger.info("Saved results * scores  to %s", out_file)


def get_datasets(qa_file_pattern):
    logger.info(f"Reading datasets usign the pattern {qa_file_pattern}")
    all_patterns = qa_file_pattern.split(",")
    all_qa_files = functools.reduce(lambda a, b: a + b, [glob.glob(pattern) for pattern in all_patterns])
    qa_file_dict = {}
    for qa_file in all_qa_files:
        dataset_name = Path(qa_file).stem.replace(".", "-")
        dataset = list(parse_qa_csv_file(qa_file)) if qa_file.endswith(".csv") else list(parse_qa_json_file(qa_file))
        questions, question_answers = [], []
        for ds_item in dataset:
            question, answers = ds_item
            questions.append(question)
            question_answers.append(answers)
        qa_file_dict[dataset_name] = (questions, question_answers)
        logger.info(f"{dataset_name}:{' ' * (20 - len(dataset_name))}{len(questions)} items")

    return qa_file_dict
