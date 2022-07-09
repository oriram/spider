#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool to get dense results and validate them
"""

import argparse
import functools
import json
import logging
import os
import time
from pathlib import Path
import numpy as np

import wandb

from retriever_utils import get_datasets, load_passages, validate, save_results
from dpr.options import print_args

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

RECALL_FILE_NAME = "recall_at_k.csv"
RESULTS_FILE_NAME = "results.json"


def read_results_file(file_path):
    logger.info(f"Reading file {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)


def get_top_ids_and_scores(d1, d2, lambda_val, n_docs=100):
    # hits = {d["id"]: d["has_answer"] for d in d1["ctxs"]}
    # hits.update({d["id"]: d["has_answer"] for d in d2["ctxs"]})
    d1 = {d["id"]: float(d["score"]) for d in d1["ctxs"]}
    d2 = {d["id"]: float(d["score"]) for d in d2["ctxs"]}

    min_d1 = min(d1.values())
    min_d2 = min(d2.values())
    d = {psg_id: d1[psg_id] + lambda_val * d2[psg_id] for psg_id in d1 if psg_id in d2}
    d.update({psg_id: d1[psg_id] + lambda_val * min_d2 for psg_id in d1 if psg_id not in d2})
    d.update({psg_id: min_d1 + lambda_val * d2[psg_id] for psg_id in d2 if psg_id not in d1})

    d = list(sorted(d.items(), key=lambda item: item[1], reverse=True))[:n_docs]
    ids = [item[0] for item in d]
    scores = [item[1] for item in d]
    # hits = [hits[psg_id] for psg_id in ids]
    return ids, scores


def main(args):
    config = vars(args)

    all_passages = load_passages(args.ctx_file)
    assert len(all_passages) > 0, "No passages data found. Please specify ctx_file param properly."

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=config
        )

    if os.path.isfile(os.path.join(args.first_results, RESULTS_FILE_NAME)):
        assert os.path.isfile(os.path.join(args.second_results, RESULTS_FILE_NAME))
        dirs_1, dirs_2 = [args.first_results], [args.second_results]
    else:
        dirs_1 = [os.path.join(args.first_results, path) for path in os.listdir(args.first_results)]
        dirs_1 = filter(os.path.isdir, dirs_1)
        dirs_2 = [os.path.join(args.second_results, path) for path in os.listdir(args.second_results)]
        dirs_2 = filter(os.path.isdir, dirs_2)

    dirs_1 = {Path(dir_path).stem: dir_path for dir_path in dirs_1}
    dirs_2 = {Path(dir_path).stem: dir_path for dir_path in dirs_2}

    assert set(dirs_1.keys()) == set(dirs_2.keys()), f"First set of datasets: {dirs_1.keys()}, second: {dirs_2.keys()}"
    logger.info(f"Dataset list: {list(dirs_1.keys())}")

    for dataset_name in dirs_1:
        logger.info("*" * 40)
        logger.info(f"Working on dataset {dataset_name}")
        dataset_output_dir = os.path.join(args.output_dir, dataset_name)
        out_file = os.path.join(dataset_output_dir, RESULTS_FILE_NAME)
        if os.path.exists(out_file):
            logger.info(f"Skipping dataset '{dataset_name}' as it already exists")
            continue
        os.makedirs(dataset_output_dir, exist_ok=True)
        first_results_file = os.path.join(dirs_1[dataset_name], RESULTS_FILE_NAME)
        second_results_file = os.path.join(dirs_2[dataset_name], RESULTS_FILE_NAME)

        results_1 = read_results_file(first_results_file)
        results_2 = read_results_file(second_results_file)
        assert len(results_1) == len(results_2)

        if args.lambda_max is None:
            lambdas = [args.lambda_min]
        else:
            lambdas = np.arange(args.lambda_min, args.lambda_max, args.lambda_step)
        match_type = "regex" if "curated" in dataset_name else args.match

        best_value, best_results, best_lambda = -1.0, None, None
        for lambda_val in lambdas:
            questions, question_answers, entities = [], [], []
            top_ids_and_scores = []
            for d1, d2 in zip(results_1, results_2):
                assert d1["question"] == d2["question"]
                questions.append(d1["question"])
                question_answers.append(d1["answers"])
                entities.append(None)
                ids, scores = get_top_ids_and_scores(d1, d2, lambda_val, args.n_docs)
                top_ids_and_scores.append((ids, scores))

            out_file = os.path.join(dataset_output_dir, f"recall_at_k_lambda_{lambda_val}.csv")
            questions_doc_hits, top_k_hits = validate(
                dataset_name,
                all_passages,
                question_answers,
                top_ids_and_scores,
                args.num_threads,
                match_type,
                out_file,
                use_wandb=False,
                output_recall_at_k=True,
                log=False
            )

            recall_at_k = top_k_hits[args.k_to_optimize - 1]
            logger.info(f"Lambda={lambda_val}, recall_at_{args.k_to_optimize}: {recall_at_k}")
            if recall_at_k > best_value:
                logger.info(f"Best lambda so far!")
                best_lambda = lambda_val
                best_value = recall_at_k
                best_results = top_ids_and_scores

        top_ids_and_scores = best_results

        with open(os.path.join(dataset_output_dir, "best_lambda.txt"), "w") as f:
            f.write(str(best_lambda) + "\n")

        out_file = os.path.join(dataset_output_dir, RECALL_FILE_NAME)
        questions_doc_hits = validate(
            dataset_name,
            all_passages,
            question_answers,
            top_ids_and_scores,
            args.num_threads,
            match_type,
            out_file,
            use_wandb=use_wandb,
            output_recall_at_k=False,
            log=True
        )

        out_file = os.path.join(dataset_output_dir, RESULTS_FILE_NAME)
        save_results(
            all_passages,
            questions,
            question_answers,
            entities,
            top_ids_and_scores,
            questions_doc_hits,
            out_file,
            output_no_text=args.output_no_text
        )

    if use_wandb:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--first_results",
        required=True,
        type=str,
        help="A file or a directory with sub-directories with json result files from a retriever",
    )
    parser.add_argument(
        "--second_results",
        required=True,
        type=str,
        help="A file or a directory with sub-directories with json result files from a retriever",
    )
    parser.add_argument(
        "--ctx_file",
        required=True,
        type=str,
        default=None,
        help="All passages file in the tsv format: id \\t passage_text \\t title",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="output .tsv file path to write results to ",
    )
    parser.add_argument(
        "--lambda_min",
        required=True,
        type=float,
    )
    parser.add_argument(
        "--lambda_max",
        default=None,
        type=float,
        help="The maximum lambda value to consider. None if only one value is need to be considered"
    )
    parser.add_argument(
        "--lambda_step",
        default=None,
        type=float,
        help="The lambda jumps value"
    )
    parser.add_argument("--k_to_optimize", type=int, default=100, help="We optimize the recall@k (w.r.t lambda)")
    parser.add_argument(
        "--n-docs", type=int, default=100, help="Amount of top docs to return"
    )
    parser.add_argument(
        "--match",
        type=str,
        default="string",
        choices=["regex", "string"],
        help="Answer matching logic type",
    )
    parser.add_argument("--output_no_text", action="store_true")
    parser.add_argument(
        "--num_threads",
        type=int,
        default=16,
        help="Number of parallel processes to validate results",
    )
    # wandb params
    parser.add_argument(
        "--no_wandb",
        action="store_true",
    )
    parser.add_argument(
        "--wandb_project",
        default="retrieval",
        type=str
    )
    parser.add_argument(
        "--wandb_name",
        default="spider-eval",
        type=str,
        help="Experiment name for W&B"
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print_args(args, args.output_dir)
    main(args)
