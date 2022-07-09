#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool that produces embeddings for a large documents base based on the pretrained ctx & question encoders
 Supposed to be used in a 'sharded' way to speed up the process.
"""
import glob
import os
import pathlib
import ntpath

import argparse
import csv
import logging
import pickle
import time
from typing import List, Tuple

import numpy as np
import torch
import transformers
from torch import nn
from torch.cuda.amp import autocast
from transformers import PreTrainedTokenizerBase

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
    move_to_device,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


def gen_ctx_vectors(
    # ctx_rows: List[Tuple[object, str, str]],
    ctx_rows: List[Tuple[object, List[int]]],
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[object, np.array]]:
    n = len(ctx_rows)
    bsz = args.batch_size
    total = 0
    results = [None] * n

    logger.info("Sorting contexts")
    sorted_indices = list(reversed(np.argsort([len(input_ids) for _, input_ids in ctx_rows])))

    logger.info("Generating embeddings...")
    start_time = time.time()
    for j, batch_start in enumerate(range(0, n, bsz)):
        batch_indices = sorted_indices[batch_start : batch_start + bsz]
        batch_rows = [ctx_rows[ind] for ind in batch_indices]
        batch_ids, batch_inputs = zip(*batch_rows)
        batch_ids = list(batch_ids)
        batch_inputs = [(ex_ids, None) for ex_ids in batch_inputs]
        encoded = tokenizer.batch_encode_plus(batch_inputs, is_pretokenized=True, add_special_tokens=False,
                                              padding="longest", return_tensors="pt", return_attention_mask=True)

        input_ids = encoded["input_ids"].to(args.device)
        attention_mask = encoded["attention_mask"].to(args.device)
        token_type_ids = torch.zeros_like(input_ids)

        with autocast(enabled=args.fp16), torch.no_grad():
            _, out, _ = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        out = out.cpu()

        assert len(batch_ids) == out.size(0)

        total += len(batch_ids)

        for i, ind in enumerate(batch_indices):
            assert results[ind] is None
            results[ind] = (batch_ids[i], out[i].view(-1).numpy())

        if total % 10 == 0:
            logger.info(f"Encoded {total} passages, took {time.time()-start_time:.1f} seconds")
    logger.info(f"Done. Took {(time.time()-start_time)/60:.1f} minutes")

    return results


def convert_dict_ctxs_to_list(tokenizer, rows, max_sequence_length=240):
    new_rows = []
    for ctx_id, (title_input_ids, ctx_input_ids) in rows.items():
        encoded = tokenizer.encode(
            title_input_ids,
            text_pair=ctx_input_ids,
            add_special_tokens=True,
            max_length=max_sequence_length,
            pad_to_max_length=False,
            truncation=True
        )
        new_rows.append((str(ctx_id), encoded))
    new_rows = sorted(new_rows, key=lambda x: int(x[0]))
    return new_rows


def main(args):
    # if model file is specified, encoder parameters from saved state should be used for initialization
    saved_state = None
    if args.model_file and os.path.exists(args.model_file):
        saved_state = load_states_from_checkpoint(args.model_file)
        set_encoder_params_from_state(saved_state.encoder_params, args)
    else:
        logger.info(f"A model_file was not passed. Model is initialized from {args.pretrained_model_cfg}")

    os.makedirs(args.output_dir, exist_ok=True)
    print_args(args, args.output_dir)

    tensorizer, encoder, _ = init_biencoder_components(
        args.encoder_model_type, args, inference_only=True
    )
    tokenizer = tensorizer.tokenizer

    encoder = encoder.get_context_encoder()

    encoder, _ = setup_for_distributed_mode(
        encoder,
        None,
        args.device,
        args.n_gpu,
        args.local_rank,
        args.fp16,
        args.fp16_opt_level,
    )
    encoder.eval()

    # load weights from the model file
    if saved_state is not None:
        model_to_load = get_model_obj(encoder)
        logger.info("Loading saved model state ...")
        logger.debug("saved model keys =%s", saved_state.model_dict.keys())

        prefix_len = len("model.") if args.weight_sharing else len("ctx_model.")
        ctx_state = {
            key[prefix_len:]: value
            for (key, value) in saved_state.model_dict.items()
            if key.startswith("ctx_model.") or key.startswith("model.")
        }
        model_to_load.load_state_dict(ctx_state, strict=False)  # TODO: strict=True?

    transformers.logging.set_verbosity_error()

    input_files = glob.glob(args.input_files)
    logger.info(f"Processing {len(input_files)} files.")
    total_num_psgs = 0
    for i, input_file in enumerate(input_files):
        logger.info(f"Processing file {i+1}/{len(input_files)}: {input_file}")
        out_file = os.path.join(args.output_dir, ntpath.basename(input_file).replace("tokenized", "wikipedia"))

        if os.path.exists(out_file):
            logger.info(f"Output file already exists at {out_file}")
            continue

        with open(input_file, "rb") as f:
            rows = pickle.load(f)

        if isinstance(rows, dict):
            rows = convert_dict_ctxs_to_list(tokenizer, rows, args.sequence_length)

        data = gen_ctx_vectors(rows, encoder, tokenizer)

        logger.info("Writing results to %s" % out_file)
        with open(out_file, mode="wb") as f:
            pickle.dump(data, f)
        total_num_psgs += len(data)
        logger.info(f"Total passages processed {total_num_psgs} from {i+1} files. Written to {out_file}")
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    parser.add_argument(
        "--input_files",
        type=str,
        required=True,
        help="Path to passages set .pkl file"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        default=None,
        help="output .tsv file path to write results to ",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for the passage encoder forward pass",
    )
    args = parser.parse_args()

    setup_args_gpu(args)

    main(args)
