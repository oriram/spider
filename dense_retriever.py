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
import glob
import logging
import os
import pickle
import time
from multiprocessing.pool import Pool
from typing import List, Tuple, Iterator, Optional

import numpy as np
import torch
import wandb
from torch import Tensor as T
from torch import nn

from dpr.indexer.faiss_indexers import (
    DenseIndexer,
    DenseHNSWFlatIndexer,
    DenseFlatIndexer,
)
from dpr.models import init_biencoder_components
from dpr.options import (
    add_encoder_params,
    setup_args_gpu,
    print_args,
    set_encoder_params_from_state,
    add_tokenizer_params,
    add_cuda_params,
)
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
)
from retriever_utils import get_datasets, load_passages, validate, save_results

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)

RECALL_FILE_NAME = "recall_at_k.csv"
RESULTS_FILE_NAME = "results.json"


class DenseRetriever(object):
    """
    Does passage retrieving over the provided index and question encoder
    """

    def __init__(
        self,
        question_encoder: nn.Module,
        batch_size: int,
        tensorizer: Tensorizer,
        index: Optional[DenseIndexer],
    ):
        self.question_encoder = question_encoder
        self.batch_size = batch_size
        self.tensorizer = tensorizer
        self.index = index

    def generate_question_vectors(self, questions: List[str]) -> T:
        n = len(questions)
        bsz = self.batch_size
        query_vectors = []

        self.question_encoder.eval()

        with torch.no_grad():
            for j, batch_start in enumerate(range(0, n, bsz)):

                batch_token_tensors = [
                    self.tensorizer.text_to_tensor(q)
                    for q in questions[batch_start : batch_start + bsz]
                ]

                q_ids_batch = torch.stack(batch_token_tensors, dim=0).cuda()
                q_seg_batch = torch.zeros_like(q_ids_batch).cuda()
                q_attn_mask = self.tensorizer.get_attn_mask(q_ids_batch)
                _, out, _ = self.question_encoder(q_ids_batch, q_seg_batch, q_attn_mask)

                query_vectors.extend(out.cpu().split(1, dim=0))

                if len(query_vectors) % 100 == 0:
                    logger.info("Encoded queries %d", len(query_vectors))

        query_tensor = torch.cat(query_vectors, dim=0)

        logger.info("Total encoded queries tensor %s", query_tensor.size())

        assert query_tensor.size(0) == len(questions)
        return query_tensor

    def get_top_docs(
        self, query_vectors: np.array, top_docs: int = 100
    ) -> List[Tuple[List[object], List[float]]]:
        """
        Does the retrieval of the best matching passages given the query vectors batch
        :param query_vectors:
        :param top_docs:
        :return:
        """
        time0 = time.time()
        results = self.index.search_knn(query_vectors, top_docs)
        logger.info("index search time: %f sec.", time.time() - time0)
        return results


def iterate_encoded_files(vector_files: list) -> Iterator[Tuple[object, np.array]]:
    for i, file in enumerate(vector_files):
        logger.info("Reading file %s", file)
        with open(file, "rb") as reader:
            doc_vectors = pickle.load(reader)
            for doc in doc_vectors:
                db_id, doc_vector = doc
                yield db_id, doc_vector


def iterate_file(input_path):
    logger.info(f"Iterating over file {input_path} to speed up index creation")
    with open(input_path, "rb") as f:
        pickle.load(f)


def iterate_over_all_files(input_paths):
    with Pool(os.cpu_count()) as p:
        p.starmap(iterate_file, [(input_path, ) for input_path in input_paths])


def main(args):
    # if model file is specified, encoder parameters from saved state should be used for initialization
    saved_state = None
    if args.model_file and os.path.exists(args.model_file):
        saved_state = load_states_from_checkpoint(args.model_file)
        set_encoder_params_from_state(saved_state.encoder_params, args)
    else:
        logger.info(f"A model_file was not passed. Model is initialized from {args.pretrained_model_cfg}")

    config = vars(args)

    tensorizer, encoder, _ = init_biencoder_components(
        args.encoder_model_type, args, inference_only=True
    )

    encoder = encoder.get_question_encoder()

    encoder, _ = setup_for_distributed_mode(
        encoder, None, args.device, args.n_gpu, args.local_rank, args.fp16
    )
    encoder.eval()

    # load weights from the model file
    model_to_load = get_model_obj(encoder)
    if saved_state is not None:
        logger.info("Loading saved model state ...")

        prefix_len = len("model.") if args.weight_sharing else len("question_model.")
        question_encoder_state = {
            key[prefix_len:]: value
            for (key, value) in saved_state.model_dict.items()
            if key.startswith("question_model.") or key.startswith("model.")
        }
        model_to_load.load_state_dict(question_encoder_state, strict=False)  # TODO: strict=True?
    vector_size = model_to_load.get_out_size()
    logger.info("Encoder vector_size=%d", vector_size)

    # get questions & answers
    qa_file_dict = get_datasets(args.qa_file)

    all_passages = load_passages(args.ctx_file)
    if len(all_passages) == 0:
        raise RuntimeError(
            "No passages data found. Please specify ctx_file param properly."
        )

    # Create or load retriever
    if args.hnsw_index:
        index = DenseHNSWFlatIndexer(vector_size, args.index_buffer)
    else:
        index = DenseFlatIndexer(vector_size, args.index_buffer, num_threads=args.num_threads)

    retriever = DenseRetriever(encoder, args.batch_size, tensorizer, index)

    # index all passages
    ctx_files_pattern = args.encoded_ctx_file
    input_paths = glob.glob(ctx_files_pattern)
    logger.info(f"Number of index files found: {len(input_paths)}")
    # We now iterate all files in parallel to speed up loading time
    if args.iterate_files_before_index:
        iterate_over_all_files(input_paths)

    index_path = args.index_path
    start_time = time.time()
    if args.save_or_load_index and (
        os.path.exists(index_path) or os.path.exists(index_path + ".index.dpr")
    ):
        logger.info(f"Using index from: {index_path}")
        retriever.index.deserialize_from(index_path)
        logger.info(f"Done. Took {time.time() - start_time:.0f} seconds to load.")
    else:
        logger.info("Reading all passages data from files: %s", input_paths)
        retriever.index.index_data(input_paths)
        logger.info(f"Done. Took {time.time() - start_time:.0f} seconds to build index.")

        if args.save_or_load_index:
            start_time = time.time()
            os.makedirs(index_path)
            retriever.index.serialize(index_path)
            logger.info(f"Done. Took {time.time() - start_time:.0f} seconds to serialize index.")

    use_wandb = not args.no_wandb
    if use_wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=config
        )

    # get top k results
    for dataset_name, (questions, question_answers) in qa_file_dict.items():
        logger.info("*" * 40)
        logger.info(f"Working on dataset {dataset_name}")
        dataset_output_dir = os.path.join(args.output_dir, dataset_name)
        out_file = os.path.join(dataset_output_dir, RECALL_FILE_NAME)
        if os.path.exists(out_file):
            logger.info(f"Skipping dataset '{dataset_name}' as it already exists")
            continue
        os.makedirs(dataset_output_dir, exist_ok=True)

        questions_tensor = retriever.generate_question_vectors(questions)
        top_ids_and_scores = retriever.get_top_docs(questions_tensor.numpy(), args.n_docs)

        match_type = "regex" if "curated" in dataset_name else args.match

        # out_file = os.path.join(args.output_dir, RECALL_FILE_NAME)
        questions_doc_hits = validate(
            dataset_name,
            all_passages,
            question_answers,
            top_ids_and_scores,
            args.validation_workers,
            match_type,
            out_file,
            use_wandb=use_wandb
        )

        out_file = os.path.join(dataset_output_dir, RESULTS_FILE_NAME)
        save_results(
            all_passages,
            questions,
            question_answers,
            top_ids_and_scores,
            questions_doc_hits,
            out_file,
            output_no_text=args.output_no_text
        )

    if use_wandb:
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    parser.add_argument(
        "--qa_file",
        required=True,
        type=str,
        default=None,
        help="Question and answers file of the format: question \\t ['answer1','answer2', ...]",
    )
    parser.add_argument(
        "--ctx_file",
        required=True,
        type=str,
        default=None,
        help="All passages file in the tsv format: id \\t passage_text \\t title",
    )
    parser.add_argument(
        "--encoded_ctx_file",
        type=str,
        default=None,
        help="Glob path to encoded passages (from generate_dense_embeddings tool)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="output .tsv file path to write results to ",
    )
    parser.add_argument(
        "--match",
        type=str,
        default="string",
        choices=["regex", "string"],
        help="Answer matching logic type",
    )
    parser.add_argument(
        "--n-docs", type=int, default=100, help="Amount of top docs to return"
    )
    parser.add_argument("--output_no_text", action="store_true")
    parser.add_argument("--scoring", choices=["dpr", "max"], default="dpr")
    parser.add_argument(
        "--validation_workers",
        type=int,
        default=16,
        help="Number of parallel processes to validate results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for question encoder forward pass",
    )
    parser.add_argument(
        "--index_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--index_buffer",
        type=int,
        default=50000,
        help="Temporal memory data buffer size (in samples) for indexer",
    )
    parser.add_argument(
        "--hnsw_index",
        action="store_true",
        help="If enabled, use inference time efficient HNSW index",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        help="Number of threads to use while searching in the index"
    )
    parser.add_argument(
        "--save_or_load_index", action="store_true", help="If enabled, save index"
    )
    parser.add_argument(
        "--iterate_files_before_index",
        action="store_true",
        help="Whether to iterate all files in parallel before building index (helps reading them faster)"
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

    setup_args_gpu(args)
    os.makedirs(args.output_dir, exist_ok=True)
    assert not os.path.exists(os.path.join(args.output_dir, RECALL_FILE_NAME))
    assert not os.path.exists(os.path.join(args.output_dir, RESULTS_FILE_NAME))
    print_args(args, args.output_dir)
    main(args)
