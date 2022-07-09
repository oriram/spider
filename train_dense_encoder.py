#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
 Pipeline to train DPR Biencoder
"""

import argparse
import glob
import json
import logging
import math
import os
import pickle
import random
import shutil
import time
from multiprocessing.pool import Pool

import numpy as np
import transformers

import wandb
import torch

from typing import Tuple
from torch import nn
import torch.nn.functional as F
from torch import Tensor as T
from torch.cuda.amp import autocast, GradScaler

from dpr.models import init_biencoder_components
from dpr.models.biencoder import BiEncoder, BiEncoderNllLoss, BiEncoderBatch
from dpr.options import (
    add_encoder_params,
    add_training_params,
    setup_args_gpu,
    set_seed,
    print_args,
    get_encoder_params_state,
    add_tokenizer_params,
    set_encoder_params_from_state,
)
from dpr.utils.data_utils import (
    ShardedDataIterator,
    read_data_from_json_files,
    Tensorizer,
)
from dpr.utils.dist_utils import all_gather_list
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    move_to_device,
    get_schedule_linear,
    CheckpointState,
    get_model_file,
    get_model_obj,
    load_states_from_checkpoint,
    convert_weight_sharing_in_saved_state
)
from query_transformations import create_query_transformation

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


class BiEncoderTrainer(object):
    """
    BiEncoder training pipeline component. Can be used to initiate or resume training and validate the trained model
    using either binary classification's NLL loss or average rank of the question's gold passages across dataset
    provided pools of negative passages. For full IR accuracy evaluation, please see generate_dense_embeddings.py
    and dense_retriever.py CLI tools.
    """

    def __init__(self, args):
        self.args = args
        self.shard_id = args.local_rank if args.local_rank != -1 else 0
        self.distributed_factor = args.distributed_world_size or 1

        logger.info("***** Initializing components for training *****")

        # if model file is specified, encoder parameters from saved state should be used for initialization
        model_file = get_model_file(self.args, self.args.checkpoint_file_name)
        saved_state = None
        if model_file:
            saved_state = load_states_from_checkpoint(model_file)
            set_encoder_params_from_state(saved_state.encoder_params, args)
            if args.weight_sharing and args.cancel_weight_sharing:
                saved_state = convert_weight_sharing_in_saved_state(saved_state)
                args.weight_sharing = False
                logger.info(f"Overriding args parameter value from checkpoint state. Param = weight_sharing, value = False")

        tensorizer, model, optimizer = init_biencoder_components(
            args.encoder_model_type, args
        )

        model, optimizer = setup_for_distributed_mode(
            model,
            optimizer,
            args.device,
            args.n_gpu,
            args.local_rank,
            args.fp16,
            args.fp16_opt_level,
        )
        self.biencoder = model
        self.optimizer = optimizer
        self.tensorizer = tensorizer
        self.scaler = None
        self.global_step = 0
        self.start_epoch = 0
        self.start_batch = 0
        self.curr_epoch = -1
        self.scheduler_state = None
        self.best_validation_result = None
        self.validation_file_path = os.path.join(args.output_dir, "eval_results.csv")
        with open(self.validation_file_path, "w") as f:
            f.write("epoch,iteration,loss,accuracy,average_rank,mrr\n")
        self.best_cp_name = None
        self.best_cp_document_file = os.path.join(args.output_dir, "best_cp.txt")
        self.load_only_model = args.load_only_model
        if saved_state:
            self._load_saved_state(saved_state)
        self.use_wandb = not args.no_wandb

        self.passages = self.get_all_tokenized_passages(self.args.tokenized_passages) \
            if (self.args.tokenized_passages is not None) else None

        self.query_transformation = create_query_transformation(args, self.tensorizer.tokenizer) \
            if args.pretraining else None

    def get_data_iterator(
        self,
        path: str,
        batch_size: int,
        shuffle=True,
        shuffle_seed: int = 0,
        offset: int = 0,
        upsample_rates: list = None,
        num_examples: list = None
    ) -> ShardedDataIterator:
        data_files = []
        for pattern in path.split(","):
            data_files.extend(glob.glob(pattern))
        data = read_data_from_json_files(data_files, upsample_rates, num_examples)

        # filter those without positive ctx
        data = [r for r in data if len(r["positive_ctxs"]) > 0]
        logger.info("Total cleaned data size: {}".format(len(data)))

        return ShardedDataIterator(
            data,
            shard_id=self.shard_id,
            num_shards=self.distributed_factor,
            batch_size=batch_size,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            offset=offset,
            strict_batch_size=True,  # this is not really necessary, one can probably disable it
        )

    @staticmethod
    def _get_single_file(path):
        if path.endswith(".jsonl"):
            with open(path, "r") as f:
                return [json.loads(line) for line in f]
        elif path.endswith(".pkl"):
            with open(path, "rb") as f:
                return pickle.load(f)
        raise ValueError

    def get_all_tokenized_passages(self, tokenized_passages_pattern):
        paths = glob.glob(tokenized_passages_pattern)
        logger.info(f"Getting tokenized passages from {len(paths)} files, the pattern is {tokenized_passages_pattern}")
        start_time = time.time()
        with Pool(len(paths)) as p:
            all_files = p.starmap(self._get_single_file, [(path, ) for path in paths])
        all_passages = {}
        for pickle_file in all_files:
            all_passages.update(pickle_file)
        # Assert all passages are present:
        max_psg_id = max(all_passages.keys())
        assert all(psg_id in all_passages for psg_id in range(1, max_psg_id))
        duration = (time.time() - start_time)/60
        logger.info(f"Finished reading and processing all passages.. Took {duration:0.1f} minutes")
        return all_passages

    def get_all_pretraining_data(self, data_file_pattern):
        paths = glob.glob(data_file_pattern)
        logger.info(f"Getting training data from {len(paths)} files, the pattern is {data_file_pattern}")
        start_time = time.time()
        with Pool(len(paths)) as p:
            all_files = p.starmap(self._get_single_file, [(path,) for path in paths])
        all_data = []
        for file in all_files:
            all_data.extend(file)
        duration = (time.time() - start_time) / 60
        logger.info(f"Finished reading and processing training data.. Took {duration:0.1f} minutes")
        return all_data

    def get_pretraining_data_iterator(
        self,
        data_file_pattern: str,
        batch_size: int,
        shuffle=True,
        shuffle_seed: int = 0,
        offset: int = 0,
    ):
        data = self.get_all_pretraining_data(data_file_pattern)
        return ShardedDataIterator(
            data,
            shard_id=self.shard_id,
            num_shards=self.distributed_factor,
            batch_size=batch_size,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            offset=offset,
            strict_batch_size=True,  # this is not really necessary, one can probably disable it
        )

    def run_train(
        self,
    ):
        args = self.args
        upsample_rates = None
        num_examples = None
        if args.train_files_upsample_rates is not None:
            upsample_rates = eval(args.train_files_upsample_rates)
        if args.train_files_num_examples is not None:
            num_examples = eval(args.train_files_num_examples)

        train_iterator = self.get_data_iterator(
            args.train_file,
            args.batch_size,
            shuffle=True,
            shuffle_seed=args.seed,
            offset=self.start_batch,
            upsample_rates=upsample_rates,
            num_examples=num_examples,
        )

        logger.info("  Total iterations per epoch=%d", train_iterator.max_iterations)
        updates_per_epoch = (
            train_iterator.max_iterations // args.gradient_accumulation_steps
        )
        total_updates = (
            max(updates_per_epoch * (args.num_train_epochs - self.start_epoch - 1), 0)
            + (train_iterator.max_iterations - self.start_batch)
            // args.gradient_accumulation_steps
        )
        logger.info(" Total updates=%d", total_updates)
        warmup_steps = args.warmup_steps
        scheduler = get_schedule_linear(self.optimizer, warmup_steps, total_updates)

        if self.scheduler_state:
            logger.info("Loading scheduler state %s", self.scheduler_state)
            scheduler.load_state_dict(self.scheduler_state)

        # eval_step = math.ceil(updates_per_epoch / args.eval_per_epoch)
        # logger.info("  Eval step = %d", eval_step)
        logger.info("***** Training *****")
        
        self.scaler = GradScaler() if args.fp16 else None
        if self.use_wandb:
            wandb.watch(self.biencoder)

        for epoch in range(self.start_epoch, int(args.num_train_epochs)):
            self.curr_epoch = epoch
            logger.info(f"***** Epoch {epoch}/{args.num_train_epochs} *****")
            self._train_epoch(scheduler, epoch, train_iterator)

        if args.local_rank in [-1, 0]:
            logger.info(
                "Training finished. Best validation checkpoint %s", self.best_cp_name
            )

        if args.no_eval:
            self.best_cp_name = self._save_checkpoint(scheduler, epoch, 0)

        if not args.keep_all_checkpoints:
            os.rename(self.best_cp_name, os.path.join(args.output_dir, "best_cp"))
            logger.info("Deleting all unnecessary checkpoints..")
            shutil.rmtree(os.path.join(args.output_dir, "checkpoints"))

    def run_pretraining(
            self,
    ):
        args = self.args
        upsample_rates = None
        num_examples = None
        # if args.train_files_upsample_rates is not None:
        #     upsample_rates = eval(args.train_files_upsample_rates)
        # if args.train_files_num_examples is not None:
        #     num_examples = eval(args.train_files_num_examples)

        train_iterator = self.get_pretraining_data_iterator(
            args.train_file,
            args.batch_size,
            shuffle=True,
            shuffle_seed=args.seed,
            offset=self.start_batch,
        )

        total_updates = args.update_steps
        updates_per_epoch = train_iterator.max_iterations // args.gradient_accumulation_steps
        num_epochs = math.ceil(total_updates / updates_per_epoch) - self.start_epoch
        logger.info(" Total updates=%d", total_updates)
        logger.info(" Num epochs=%d", num_epochs)
        warmup_steps = args.warmup_steps
        scheduler = get_schedule_linear(self.optimizer, warmup_steps, total_updates)

        if self.scheduler_state:
            logger.info("Loading scheduler state %s", self.scheduler_state)
            scheduler.load_state_dict(self.scheduler_state)

        # eval_step = math.ceil(updates_per_epoch / args.eval_per_epoch)
        # logger.info("  Eval step = %d", eval_step)
        logger.info("***** Training *****")

        self.scaler = GradScaler() if args.fp16 else None
        if self.use_wandb:
            wandb.watch(self.biencoder)

        for epoch in range(self.start_epoch, num_epochs):
            self.curr_epoch = epoch
            logger.info(f"***** Epoch {epoch}/{num_epochs} *****")
            self._train_epoch(scheduler, epoch, train_iterator)

        if args.local_rank in [-1, 0]:
            logger.info(
                "Training finished. Best validation checkpoint %s", self.best_cp_name
            )

    def validate_and_save(self, epoch: int, iteration: int, scheduler, save=True, eval=True):
        args = self.args
        if (not save) and (not eval):
            return

        # for distributed mode, save checkpoint for only one process
        save_cp = save and args.local_rank in [-1, 0]

        if eval:
            if epoch == args.val_av_rank_start_epoch:
                self.best_validation_result = None

            if epoch >= args.val_av_rank_start_epoch:
                total_loss, correct_ratio = self.validate_nll()
                av_rank, mrr, recall_at = self.validate_average_rank()
                validation_measure = mrr
            else:
                total_loss, correct_ratio = self.validate_nll()
                av_rank, mrr, recall_at = None, None, None
                validation_measure = correct_ratio

            # Log results to W&B
            d = {"eval/accuracy": correct_ratio, "eval/loss": total_loss,
                 "eval/epoch": self.curr_epoch, "eval/global_step": self.global_step}
            if av_rank is not None:
                d["eval/average_rank"] = av_rank
                d["eval/mrr"] = mrr
                d["eval/psg_recall_at_1"] = recall_at[0]
                d["eval/psg_recall_at_10"] = recall_at[9]
                d["eval/psg_recall_at_100"] = recall_at[99]

            if self.use_wandb:
                wandb.log(d)
            # Log results to a file
            with open(self.validation_file_path, "a") as f:
                f.write(f"{epoch},{iteration},{total_loss},{correct_ratio},{av_rank},{mrr}\n")
        else:
            validation_measure = None

        if save_cp:
            cp_name = self._save_checkpoint(scheduler, epoch, iteration)
            logger.info("Saved checkpoint to %s", cp_name)

            if validation_measure is not None and validation_measure > (self.best_validation_result or validation_measure - 1):
                self.best_validation_result = validation_measure
                self.best_cp_name = cp_name
                logger.info("New Best validation checkpoint %s", cp_name)
                with open(self.best_cp_document_file, "w") as f:
                    f.write(self.best_cp_name + "\n")

    def validate_nll(self) -> Tuple[float, float]:
        logger.info("NLL validation ...")
        start_time = time.time()
        args = self.args
        self.biencoder.eval()
        data_iterator = self.get_data_iterator(
            args.dev_file, args.dev_batch_size, shuffle=False
        )

        total_loss = 0.0
        start_time = time.time()
        total_correct_predictions = 0
        num_hard_negatives = args.val_hard_negatives if args.val_hard_negatives is not None else args.hard_negatives
        num_other_negatives = args.other_negatives
        log_result_step = args.log_batch_step
        batches = 0
        for i, samples_batch in enumerate(data_iterator.iterate_data()):
            biencoder_input = BiEncoder.create_biencoder_input(
                samples_batch,
                self.tensorizer,
                True,
                num_hard_negatives,
                num_other_negatives,
                shuffle=False,
                passages=self.passages
            )

            with autocast(enabled=args.fp16), torch.no_grad():
                loss, correct_cnt = _do_biencoder_fwd_pass(
                    self.biencoder, biencoder_input, self.tensorizer, args
                )
            total_loss += loss.item()
            total_correct_predictions += correct_cnt
            batches += 1
            if (i + 1) % log_result_step == 0:
                logger.info(
                    "Eval step: %d , used_time=%f sec., loss=%f ",
                    i,
                    time.time() - start_time,
                    loss.item(),
                )

        total_loss = total_loss / batches
        total_samples = batches * args.dev_batch_size * self.distributed_factor
        correct_ratio = float(total_correct_predictions / total_samples)
        logger.info(
            "NLL Validation: loss = %f. correct prediction ratio  %d/%d ~  %f",
            total_loss,
            total_correct_predictions,
            total_samples,
            correct_ratio,
        )
        logger.info(f"NLL validation took {(time.time()-start_time)/60:0.1f} minutes")
        return total_loss, correct_ratio

    def validate_average_rank(self):
        """
        Validates biencoder model using each question's gold passage's rank across the set of passages from the dataset.
        It generates vectors for specified amount of negative passages from each question (see --val_av_rank_xxx params)
        and stores them in RAM as well as question vectors.
        Then the similarity scores are calculted for the entire
        num_questions x (num_questions x num_passages_per_question) matrix and sorted per quesrtion.
        Each question's gold passage rank in that  sorted list of scores is averaged across all the questions.
        :return: averaged rank number
        """
        logger.info("Average rank validation ...")
        start_time = time.time()

        args = self.args
        self.biencoder.eval()
        distributed_factor = self.distributed_factor

        data_iterator = self.get_data_iterator(
            args.dev_file, args.dev_batch_size, shuffle=False
        )

        sub_batch_size = args.val_av_rank_bsz
        sim_score_f = BiEncoderNllLoss.get_similarity_function()
        q_represenations = []
        ctx_represenations = []
        positive_idx_per_question = []

        num_hard_negatives = args.val_av_rank_hard_neg
        num_other_negatives = args.val_av_rank_other_neg

        log_result_step = args.log_batch_step

        for i, samples_batch in enumerate(data_iterator.iterate_data()):
            # samples += 1
            if len(q_represenations) > args.val_av_rank_max_qs / distributed_factor:
                break

            biencoder_input = BiEncoder.create_biencoder_input(
                samples_batch,
                self.tensorizer,
                True,
                num_hard_negatives,
                num_other_negatives,
                shuffle=False,
                passages=self.passages
            )
            total_ctxs = len(ctx_represenations)
            ctxs_ids = biencoder_input.context_ids
            ctxs_segments = biencoder_input.ctx_segments
            bsz = ctxs_ids.size(0)

            # split contexts batch into sub batches since it is supposed to be too large to be processed in one batch
            for j, batch_start in enumerate(range(0, bsz, sub_batch_size)):

                q_ids, q_segments = (
                    (biencoder_input.question_ids, biencoder_input.question_segments)
                    if j == 0
                    else (None, None)
                )

                if j == 0 and args.n_gpu > 1 and q_ids.size(0) == 1:
                    # if we are in DP (but not in DDP) mode, all model input tensors should have batch size >1 or 0,
                    # otherwise the other input tensors will be split but only the first split will be called
                    continue

                ctx_ids_batch = ctxs_ids[batch_start : batch_start + sub_batch_size]
                ctx_seg_batch = ctxs_segments[
                    batch_start : batch_start + sub_batch_size
                ]

                q_attn_mask = self.tensorizer.get_attn_mask(q_ids)
                ctx_attn_mask = self.tensorizer.get_attn_mask(ctx_ids_batch)
                with autocast(enabled=args.fp16), torch.no_grad():
                    q_dense, ctx_dense = self.biencoder(
                        q_ids,
                        q_segments,
                        q_attn_mask,
                        ctx_ids_batch,
                        ctx_seg_batch,
                        ctx_attn_mask,
                    )

                if q_dense is not None:
                    q_represenations.extend(q_dense.cpu().split(1, dim=0))

                ctx_represenations.extend(ctx_dense.cpu().split(1, dim=0))

            batch_positive_idxs = biencoder_input.is_positive
            positive_idx_per_question.extend(
                [total_ctxs + v for v in batch_positive_idxs]
            )

            if (i + 1) % log_result_step == 0:
                logger.info(
                    "Av.rank validation: step %d, computed ctx_vectors %d, q_vectors %d",
                    i,
                    len(ctx_represenations),
                    len(q_represenations),
                )

        ctx_represenations = torch.cat(ctx_represenations, dim=0)
        q_represenations = torch.cat(q_represenations, dim=0)

        logger.info(
            "Av.rank validation: total q_vectors size=%s", q_represenations.size()
        )
        logger.info(
            "Av.rank validation: total ctx_vectors size=%s", ctx_represenations.size()
        )
        logger.info(f"Creating the two matrices took {(time.time()-start_time)/60:0.1f} minutes")

        q_num = q_represenations.size(0)
        assert q_num == len(positive_idx_per_question)

        with autocast(enabled=args.fp16), torch.no_grad():
            scores = sim_score_f(q_represenations, ctx_represenations)
            values, indices = torch.sort(scores, dim=1, descending=True)  # [num_questions, num_psgs]

        rank = 0
        inverse_rank = 0
        ranks = [0] * 100
        for i, idx in enumerate(positive_idx_per_question):
            # aggregate the rank of the known gold passage in the sorted results for each question
            gold_idx = (indices[i] == idx).nonzero().item()
            rank += gold_idx
            inverse_rank += (1 / (gold_idx + 1))
            if gold_idx < len(ranks):
                ranks[gold_idx] += 1

        if distributed_factor > 1:
            # each node calcuated its own rank, exchange the information between node and calculate the "global" average rank
            # NOTE: the set of passages is still unique for every node
            eval_stats = all_gather_list([rank, q_num], max_size=100)
            for i, item in enumerate(eval_stats):
                remote_rank, remote_q_num = item
                if i != args.local_rank:
                    rank += remote_rank
                    q_num += remote_q_num

        av_rank = float(rank / q_num)
        mrr = inverse_rank / q_num
        recall_at = np.cumsum(ranks) / q_num
        logger.info(
            f"Av.rank validation: #Qs: {q_num}, Av.Rank: {av_rank:.1f}, MRR: {mrr:.2f}, "
            f"R@1: {100*recall_at[0]:.1f}, R@10: {100*recall_at[9]:.1f}, R@100: {100*recall_at[99]:.1f}"
        )
        logger.info(f"Av.rank validation took {(time.time()-start_time)/60:0.1f} minutes")
        
        return av_rank, mrr, recall_at

    def _train_epoch(
        self,
        scheduler,
        epoch: int,
        train_data_iterator: ShardedDataIterator,
    ):

        args = self.args
        rolling_train_loss = 0.0
        rolling_train_accuracy = 0.0
        epoch_loss = 0
        epoch_correct_predictions = 0

        log_result_step = args.log_batch_step
        rolling_loss_step = args.train_rolling_loss_step
        num_hard_negatives = args.hard_negatives
        num_other_negatives = args.other_negatives
        seed = args.seed
        self.biencoder.train()
        epoch_batches = train_data_iterator.max_iterations
        data_iteration = 0
        positives_pool = set() if num_hard_negatives > 1 else None
        for i, samples_batch in enumerate(
            train_data_iterator.iterate_data(epoch=epoch)
        ):

            # to be able to resume shuffled ctx- pools
            insert_titles = True
            data_iteration = train_data_iterator.get_iteration()
            random.seed(seed + epoch + data_iteration)
            if not self.args.pretraining:
                biencoder_batch = BiEncoder.create_biencoder_input(
                    samples_batch,
                    self.tensorizer,
                    insert_titles,
                    num_hard_negatives,
                    num_other_negatives,
                    shuffle=True,
                    shuffle_positives=args.shuffle_positive_ctx,
                )
            else:
                biencoder_batch, biencoder_positives = self._create_pretraining_biencoder_input(
                    samples_batch,
                    insert_titles,
                    num_hard_negatives,
                    positives_pool=positives_pool,
                    question_max_length=args.question_sequence_length
                )
                if positives_pool is not None:
                    positives_pool.update(biencoder_positives)

            with autocast(enabled=args.fp16):
                loss, correct_cnt = _do_biencoder_fwd_pass(
                    self.biencoder, biencoder_batch, self.tensorizer, args
                )

            epoch_correct_predictions += correct_cnt
            loss_item = loss.item()
            batch_accuracy = correct_cnt / len(samples_batch)
            epoch_loss += loss_item
            rolling_train_loss += loss_item
            rolling_train_accuracy += batch_accuracy

            if args.fp16:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    self.scaler.unscale_(self.optimizer)
                
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.biencoder.parameters(), args.max_grad_norm
                    )
                
                if args.fp16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                scheduler.step()
                self.biencoder.zero_grad()
                self.global_step += 1

            lr = self.optimizer.param_groups[0]["lr"]
            if self.use_wandb:
                d = {
                    "train/loss": loss_item,
                    "train/accuracy": batch_accuracy,
                    "train/lr": lr,
                    "train/epoch": epoch,
                    "train/global_step": self.global_step
                }
                wandb.log(d)

            if i % log_result_step == 0:
                logger.info(
                    "Epoch: %d: Step: %d/%d, loss=%f, lr=%f",
                    epoch,
                    data_iteration,
                    epoch_batches,
                    loss.item(),
                    lr,
                )

            if (i + 1) % rolling_loss_step == 0:
                latest_rolling_train_av_loss = rolling_train_loss / rolling_loss_step
                latest_rolling_train_av_acc = rolling_train_accuracy / rolling_loss_step * 100
                logger.info(
                    f"Train batch {self.global_step}, Avg. loss per last {rolling_loss_step} "
                    f"batches: {latest_rolling_train_av_loss}, Accuracy: {latest_rolling_train_av_acc:0.1f}%",
                )
                rolling_train_loss = 0.0
                rolling_train_accuracy = 0.0

            if args.eval_steps > 0 and self.global_step % args.eval_steps == 0:
                logger.info(
                    "Validation: Epoch: %d Step: %d",
                    epoch,
                    self.global_step,
                )
                self.validate_and_save(
                   epoch, self.global_step, scheduler, eval=(not args.no_eval), save=(not args.no_save)
                )
                self.biencoder.train()

            if self.args.update_steps is not None and self.global_step >= self.args.update_steps:
                break

        self.validate_and_save(epoch, self.global_step, scheduler, eval=(not args.no_eval), save=(not args.no_save))

        epoch_loss = (epoch_loss / epoch_batches) if epoch_batches > 0 else 0
        logger.info("Av Loss per epoch=%f", epoch_loss)
        logger.info("epoch total correct predictions=%d", epoch_correct_predictions)

    def _save_checkpoint(self, scheduler, epoch: int, offset: int) -> str:
        args = self.args
        model_to_save = get_model_obj(self.biencoder)
        cp = os.path.join(
            args.output_dir,
            args.checkpoint_file_name
            + (("." + str(epoch)) if not self.args.pretraining else "")
            + ("." + str(offset) if offset > 0 else ""),
        )

        meta_params = get_encoder_params_state(args)

        state = CheckpointState(
            model_to_save.state_dict(),
            self.optimizer.state_dict(),
            scheduler.state_dict(),
            offset,
            epoch,
            meta_params,
        )
        torch.save(state._asdict(), cp)
        logger.info("Saved checkpoint at %s", cp)
        return cp

    def _load_saved_state(self, saved_state: CheckpointState):
        model_to_load = get_model_obj(self.biencoder)
        logger.info("Loading saved model state ...")
        model_to_load.load_state_dict(
            saved_state.model_dict
        )  # set strict=False if you use extra projection

        if self.load_only_model:
            logger.info("We don't load optimizer, scheduler and step details..")
            return

        epoch = saved_state.epoch
        offset = saved_state.offset
        if offset == 0:  # epoch has been completed
            epoch += 1
        logger.info("Loading checkpoint @ batch=%s and epoch=%s", offset, epoch)

        self.start_epoch = epoch
        self.start_batch = offset

        if saved_state.optimizer_dict:
            logger.info("Loading saved optimizer state ...")
            self.optimizer.load_state_dict(saved_state.optimizer_dict)

        if saved_state.scheduler_dict:
            self.scheduler_state = saved_state.scheduler_dict

    def _create_pretraining_biencoder_input(
        self,
        samples_batch,
        insert_titles=True,
        num_hard_negatives=1,
        positives_pool=None,
        question_max_length=None
    ):
        num_total_ctxs = len(samples_batch) * (1 + num_hard_negatives)  # Each question has one positive
        questions, ctxs, positives, hard_negatives = [], [], [], []
        batch_positive_ids, batch_negative_ids = [], []
        for sample in samples_batch:
            sample_pos = sample["positive"]
            unique_pos_psgs = list(set(pos[0] for pos in sample_pos))
            assert len(unique_pos_psgs) >= 2
            random.shuffle(unique_pos_psgs)

            question_psg_index = unique_pos_psgs[0]
            _, question_psg = self.passages[question_psg_index]
            question_spans = [(pos[1], pos[2]) for pos in sample_pos if pos[0] == question_psg_index]
            question_spans = sorted(question_spans)
            question_tokens = self.query_transformation.transform_query(question_psg, question_spans)

            questions.append(self.tensorizer.text_to_tensor(question_tokens, max_length=question_max_length))
            positives.append(len(ctxs))

            positive_psg_id = unique_pos_psgs[1]
            pos_title, pos_psg = self.passages[positive_psg_id]
            ctxs.append(self.tensorizer.text_to_tensor(pos_psg, pos_title if insert_titles else None))
            batch_positive_ids.append(positive_psg_id)

            hard_negatives = sample["negative_ctxs"]
            num_hard_negatives_to_sample = min(num_hard_negatives, len(hard_negatives))
            hard_neg_psg_ids = random.sample(hard_negatives, num_hard_negatives_to_sample)
            for hard_neg_psg_id in hard_neg_psg_ids:
                hard_neg_title, hard_neg_psg = self.passages[hard_neg_psg_id]
                ctxs.append(self.tensorizer.text_to_tensor(hard_neg_psg, hard_neg_title if insert_titles else None))
                batch_negative_ids.append(hard_neg_psg_id)

        if len(ctxs) < num_total_ctxs:
            num_negatives_to_sample = num_total_ctxs - len(ctxs)
            batch_additional_negative_pool = list(positives_pool.union(batch_negative_ids))
            negative_ids = random.choices(batch_additional_negative_pool, k=num_negatives_to_sample)
            for negative_id in negative_ids:
                neg_title, neg_psg = self.passages[negative_id]
                ctxs.append(self.tensorizer.text_to_tensor(neg_psg, neg_title if insert_titles else None))

        questions_tensor = torch.cat([q.view(1, -1) for q in questions], dim=0)
        ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in ctxs], dim=0)

        question_segments = torch.zeros_like(questions_tensor)
        ctx_segments = torch.zeros_like(ctxs_tensor)

        batch = BiEncoderBatch(
            questions_tensor,
            question_segments,
            ctxs_tensor,
            ctx_segments,
            positives,
            hard_negatives,
        )
        return batch, batch_positive_ids


def _calc_loss(
    args,
    loss_function,
    local_q_vector,
    local_ctx_vectors,
    local_positive_idxs,
    local_hard_negatives_idxs: list = None,
) -> Tuple[T, bool]:
    """
    Calculates In-batch negatives schema loss and supports to run it in DDP mode by exchanging the representations
    across all the nodes.
    """
    distributed_world_size = args.distributed_world_size or 1
    if distributed_world_size > 1:
        q_vector_to_send = (
            torch.empty_like(local_q_vector).cpu().copy_(local_q_vector).detach_()
        )
        ctx_vector_to_send = (
            torch.empty_like(local_ctx_vectors).cpu().copy_(local_ctx_vectors).detach_()
        )

        global_question_ctx_vectors = all_gather_list(
            [
                q_vector_to_send,
                ctx_vector_to_send,
                local_positive_idxs,
                local_hard_negatives_idxs,
            ],
            max_size=args.global_loss_buf_sz,
        )

        global_q_vector = []
        global_ctxs_vector = []

        # ctxs_per_question = local_ctx_vectors.size(0)
        positive_idx_per_question = []
        hard_negatives_per_question = []

        total_ctxs = 0

        for i, item in enumerate(global_question_ctx_vectors):
            q_vector, ctx_vectors, positive_idx, hard_negatives_idxs = item

            if i != args.local_rank:
                global_q_vector.append(q_vector.to(local_q_vector.device))
                global_ctxs_vector.append(ctx_vectors.to(local_q_vector.device))
                positive_idx_per_question.extend([v + total_ctxs for v in positive_idx])
                hard_negatives_per_question.extend(
                    [[v + total_ctxs for v in l] for l in hard_negatives_idxs]
                )
            else:
                global_q_vector.append(local_q_vector)
                global_ctxs_vector.append(local_ctx_vectors)
                positive_idx_per_question.extend(
                    [v + total_ctxs for v in local_positive_idxs]
                )
                hard_negatives_per_question.extend(
                    [[v + total_ctxs for v in l] for l in local_hard_negatives_idxs]
                )
            total_ctxs += ctx_vectors.size(0)

        global_q_vector = torch.cat(global_q_vector, dim=0)
        global_ctxs_vector = torch.cat(global_ctxs_vector, dim=0)

    else:
        global_q_vector = local_q_vector
        global_ctxs_vector = local_ctx_vectors
        positive_idx_per_question = local_positive_idxs
        hard_negatives_per_question = local_hard_negatives_idxs

    loss, is_correct = loss_function.calc(
        global_q_vector,
        global_ctxs_vector,
        positive_idx_per_question,
        hard_negatives_per_question,
    )

    return loss, is_correct


def _do_biencoder_fwd_pass(
    model: nn.Module, input: BiEncoderBatch, tensorizer: Tensorizer, args
) -> (torch.Tensor, int):
    input = BiEncoderBatch(**move_to_device(input._asdict(), args.device))

    q_attn_mask = tensorizer.get_attn_mask(input.question_ids)
    ctx_attn_mask = tensorizer.get_attn_mask(input.context_ids)

    if model.training:
        model_out = model(
            input.question_ids,
            input.question_segments,
            q_attn_mask,
            input.context_ids,
            input.ctx_segments,
            ctx_attn_mask,
        )
    else:
        with torch.no_grad():
            model_out = model(
                input.question_ids,
                input.question_segments,
                q_attn_mask,
                input.context_ids,
                input.ctx_segments,
                ctx_attn_mask,
            )

    local_q_vector, local_ctx_vectors = model_out

    loss_function = BiEncoderNllLoss()

    loss, is_correct = _calc_loss(
        args,
        loss_function,
        local_q_vector,
        local_ctx_vectors,
        input.is_positive,
        input.hard_negatives,
    )

    is_correct = is_correct.sum().item()

    if args.n_gpu > 1:
        loss = loss.mean()
    if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps

    return loss, is_correct


def main():
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_training_params(parser)
    add_tokenizer_params(parser)
    
    # wandb params
    parser.add_argument(
        "--wandb_project",
        default="retrieval",
        type=str
    )
    parser.add_argument(
        "--wandb_name", 
        type=str,
        help="Experiment name for W&B"
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
    )

    # biencoder specific training features
    parser.add_argument(
        "--eval_steps",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--no_eval",
        action="store_true",
        help="Don't do any validation - mainly for few-shot experiments"
    )
    parser.add_argument(
        "--no_save",
        action="store_true",
        help="Don't do any validation - mainly for few-shot experiments"
    )

    parser.add_argument(
        "--global_loss_buf_sz",
        type=int,
        default=150000,
        help='Buffer size for distributed mode representations al gather operation. \
                                Increase this if you see errors like "encoded data exceeds max_size ..."',
    )

    parser.add_argument("--cancel_weight_sharing", action="store_true")
    parser.add_argument("--fix_ctx_encoder", action="store_true")
    parser.add_argument("--fix_question_encoder", action="store_true")
    parser.add_argument("--shuffle_positive_ctx", action="store_true")

    # input/output src params
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model checkpoints will be written or resumed from",
    )

    # data handling parameters
    parser.add_argument(
        "--pretraining",
        action="store_true"
    )
    parser.add_argument(
        "--tokenized_passages",
        default=None,
        type=str,
        help="Pattern to pickled tokenized files"
    )
    parser.add_argument(
        "--query_transformation",
        type=str,
        choices=[None, "mask", "prefix", "random"],
        default=None
    )
    parser.add_argument("--question_sequence_length", default=None, type=int)
    parser.add_argument("--min_window_size", default=5, type=int)
    parser.add_argument("--max_window_size", default=30, type=int)
    parser.add_argument("--keep_answer_prob", default=0.0, type=float)
    parser.add_argument("--prefix_window", action="store_true")
    parser.add_argument("--replace_with_question_token", action="store_true")
    parser.add_argument(
        "--update_steps",
        type=int,
        default=None
    )
    parser.add_argument(
        "--mask_titles_prob",
        default=0.0,
        type=float,
        help="Probability to mask out title during training"
    )
    parser.add_argument(
        "--mask_question_entity_prob",
        default=0.0,
        type=float,
        help="Probability to mask out title during training"
    )
    parser.add_argument(
        "--hard_negatives",
        default=1,
        type=int,
        help="amount of hard negative ctx per question",
    )
    parser.add_argument(
        "--val_hard_negatives",
        default=None,
        type=int,
        help="amount of hard negative ctx per question in validate_nll",
    )
    parser.add_argument(
        "--other_negatives",
        default=0,
        type=int,
        help="amount of 'other' negative ctx per question",
    )
    parser.add_argument(
        "--train_files_upsample_rates",
        type=str,
        help="list of up-sample rates per each train file. Example: [1,2,1]",
    )
    parser.add_argument(
        "--train_files_num_examples",
        type=str,
        default=None,
        help="Limit the number of data points for each dataset. -1 for taking all of the dataset. Example: [50000,-1]"
    )

    # parameters for Av.rank validation method
    parser.add_argument(
        "--val_av_rank_start_epoch",
        type=int,
        default=10000,
        help="Av.rank validation: the epoch from which to enable this validation",
    )
    parser.add_argument(
        "--val_av_rank_hard_neg",
        type=int,
        default=30,
        help="Av.rank validation: how many hard negatives to take from each question pool",
    )
    parser.add_argument(
        "--val_av_rank_other_neg",
        type=int,
        default=30,
        help="Av.rank validation: how many 'other' negatives to take from each question pool",
    )
    parser.add_argument(
        "--val_av_rank_bsz",
        type=int,
        default=128,
        help="Av.rank validation: batch size to process passages",
    )
    parser.add_argument(
        "--val_av_rank_max_qs",
        type=int,
        default=10000,
        help="Av.rank validation: max num of questions",
    )
    parser.add_argument(
        "--checkpoint_file_name",
        type=str,
        default="checkpoints/dpr_biencoder",
        help="Checkpoints file prefix",
    )
    parser.add_argument(
        "--keep_all_checkpoints",
        action="store_true"
    )
    parser.add_argument(
        "--load_only_model",
        action="store_true",
        help="Load only model weights from checkpoint. Don't load optimizer, scheduler and step details."
    )

    args = parser.parse_args()

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                args.gradient_accumulation_steps
            )
        )

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)

    setup_args_gpu(args)
    set_seed(args)
    print_args(args, args.output_dir)

    use_wandb = not args.no_wandb
    config = vars(args)

    if use_wandb:
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=config
        )

    trainer = BiEncoderTrainer(args)

    transformers.logging.set_verbosity_error()
    if args.train_file is not None:
        if not args.pretraining:
            trainer.run_train()
        else:
            trainer.run_pretraining()
    # elif args.model_file and args.dev_file:
    elif args.dev_file:
        logger.info(
            "No train files are specified. Run 2 types of validation for specified model file"
        )
        trainer.validate_and_save(save=False, epoch=0, iteration=0, scheduler=None)
    else:
        logger.warning(
            "Neither train_file or (model_file & dev_file) parameters are specified. Nothing to do."
        )

    if use_wandb:
        wandb_run.finish()


if __name__ == "__main__":
    main()
