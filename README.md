# Spider

This repository contains the code and models discussed in our paper "[Learning to Retrieve Passages without Supervision](https://arxiv.org/abs/2112.07708)" (at NAACL 2022).

Our code is based on the [repo](https://github.com/facebookresearch/DPR) released with the [DPR](https://arxiv.org/abs/2004.04906) paper.

Please note that this is the first public version of this repo, so it is likely there are some bugs.  
Feel free to report an issue :)

## Table of Contents
- [Setup](#setup)
- [Download Corpus](#download-wiki-corpus)
- [Corpus Preprocessing](#corpus-preprocessing)
- [Retrieval Evaluation](#retrieval-evaluation)
  - [Dense Retrieval](#dense-retrieval)
    - [Generate Passage Embeddings](#generate-passage-embeddings)
    - [Evaluation](#evaluation)
  - [Sparse Retrieval](#sparse-retrieval)
  - [Hybrid Retrieval](#hybrid-retrieval)
- [Pretraining](#pretraining)
- [Fine-Tuning](#fine-tuning)
- [Convert Model Checkpoints to Hugging Face and Upload to Hub](#convert-model-checkpoints-to-hugging-face-and-upload-to-hub)
- [Citation](#citation)



## Setup

To install all requirements in our repo, run:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Download Wiki Corpus

To download the Wikipedia corpus used in our paper for both pretraining and evaluation, run:

```bash
python download_data.py --resource data.wikipedia_split.psgs_w100
```
The corpus will be downloaded to `./downloads/data/wikipedia_split/psgs_w100.tsv`.


## Corpus Preprocessing

Our preprocessing is responsible for two main processes:
1. Tokenize the corpus
2. (Optional) Find all sets of recurring spans for each document - only for pretraining
 
To apply preprocessing, run:

```bash
python preprocess_corpus.py \
--corpus_path ./downloads/data/wikipedia_split/psgs_w100.tsv \
--output_dir PREPROCESSED_DATA_DIR \
--tokenizer_name bert-base-uncased \
--num_processes 64  \
[--compute_recurring_spans] \
[--min_span_length 2] \
[--max_span_length 10] 
```
Computing recurring spans is optional and only needed for pretraining. It also takes much longer (a couple of hours, depends on the number of CPUs).  
If you only wish to evaluate/fine-tune a model, you can drop the `--compute_recurring_spans` flag.

If you do wish to preprocess recurring spans, make sure you have the `en_core_web_sm` spaCy model:
```bash
python -m spacy download en_core_web_sm
```


## Retrieval Evaluation

You can use our repo to evaluate three types of models:
- Dense Models (either Spider or your own pretrained/fine-tuned model)
- Sparse Models, specifically BM25
- Hybrid Models, e.g. combine a sparse and a dense retriever into one stronger model

All of our retrieval evaluation scripts support iteration over multiple TSV/JSON datasets,
(similar to DPR formats). The datasets we used in the paper can be obtained by:
```bash
python download_data.py --resource data.retriever.qas.nq-test
python download_data.py --resource data.retriever.qas.trivia-test
python download_data.py --resource data.retriever.qas.webq-test
python download_data.py --resource data.retriever.qas.curatedtrec-test
python download_data.py --resource data.retriever.qas.squad1-test
```
The files will be downloaded to `./downloads/data/retriever/qas/*-test.csv`.

For the EntityQuestions dataset, use:
```bash
wget https://nlp.cs.princeton.edu/projects/entity-questions/dataset.zip
unzip dataset.zip
mv dataset entityqs
```
The test sets will be available at `./entityqs/test/P*.test.json`.

### Dense Retrieval

#### Generate Passage Embeddings

Embedding generation requires *tokenized passages*, see [preprocessing](#corpus-preprocessing).

```bash
python generate_dense_embeddings.py \
--encoder_model_type hf_bert \
--pretrained_model_cfg tau/spider \
[--model_file MODEL_CKPT_FILE] \
--input_file "PREPROCESSED_DATA_DIR/tokenized_*.pkl" \
--output_dir CORPUS_EMBEDDING_DIR \
--fp16 \
--do_lower_case \
--sequence_length 240 \
--batch_size BATCH_SIZE
```

Note that `--model_file` is used for checkpoint files saved in `train_dense_encoder.py`, 
so use it only for your own pretrained/fine-tuned models.  
Also, you can replace `tau/spider` with one of the following models (from Hugging Face Hub):
- DPR (trained on NQ): `facebook/dpr-ctx_encoder-single-nq-base`
- Spider-NQ: `tau/spider-nq-ctx-encoder`
- Spider-TriviaQA: `tau/spider-trivia-ctx-encoder`

#### Evaluation

After you generate the embeddings of all passages in the corpus, you can run dense retrieval evaluation:
```bash
python dense_retriever.py \
--encoder_model_type hf_bert \
--pretrained_model_cfg tau/spider \
[--model_file MODEL_CKPT_FILE] \
--qa_file glob_pattern_1.csv,glob_pattern_2.csv,...,glob_pattern_n.csv \
--ctx_file ./downloads/data/wikipedia_split/psgs_w100.tsv \
--encoded_ctx_file "CORPUS_EMBEDDING_DIR/wikipedia_passages*.pkl" \
--output_dir OUTPUT_DIR \
--n-docs 100 \
--num_threads 16 \
--batch_size 64 \
--sequence_length 240 \
[--no_wandb] \
[--wandb_project WANDB_PROJECT] \
[--wandb_name WANDB_NAME] \
[--output_no_text]
```

Note that `--model_file` is used for checkpoint files saved in `train_dense_encoder.py`, 
so use it only for your own pretrained/fine-tuned models.  
Also, you can replace `tau/spider` with one of the following models (from Hugging Face Hub):
- DPR (trained on NQ): `facebook/dpr-question_encoder-single-nq-base`
- Spider-NQ: `tau/spider-nq-question-encoder`
- Spider-TriviaQA: `tau/spider-trivia-question-encoder`

### Sparse Retrieval

Our sparse retrieval builds on [pyserini](https://github.com/castorini/pyserini), so Java 11 is required - see their [installation guide](https://github.com/castorini/pyserini/blob/master/docs/installation.md).  
If you have Java 11 installed, make sure your `JAVA_HOME` environment variable 
is set to the correct path. 
On a Linux system, the correct path might look something like `/usr/lib/jvm/java-11`.

```bash
python sparse_retriever.py \
--index_name wikipedia-dpr \
--qa_file glob_pattern_1.csv,glob_pattern_2.csv,...,glob_pattern_n.csv \
--ctx_file ./downloads/data/wikipedia_split/psgs_w100.tsv \
--output_dir OUTPUT_DIR \
--n-docs 100 \
--num_threads 16 \
[--pyserini_cache PYSERINI_CACHE] \
[--wandb_project WANDB_PROJECT] \
[--wandb_name WANDB_NAME] \
[--output_no_text]
```

### Hybrid Retrieval

Our hybrid retriever is applied on the results of two retrievers.  
Specifically, it assumes both retrievers have results for the same datasets in their directories (where each dataset has its own subdirectory).  
For example, if `./spider-results/` and `./bm25-results/` are the two directories, they may look like:  
```bash
> ls spider-results
curatedtrec-test  nq-test   squad1-test   trivia-test   webquestions-test

> ls bm25-results
curatedtrec-test  nq-test   squad1-test   trivia-test   webquestions-test  
```

In our paper we use `k=1000` (i.e. `--n-docs 1000`) for these two retrievers.  
Since the results file are quite bug, you can run `dense_retriever.py` and `sparse_retriever.py` with `--output_no_text` which is more disk-efficient.

```bash
python hybrid_retriever.py \
--first_results FIRST_RETRIEVER_OUTPUT_DIR \
--second_results SECOND_RETRIEVER_OUTPUT_DIR \
--ctx_file ./downloads/data/wikipedia_split/psgs_w100.tsv \
--output_dir OUTPUT_DIR \
--n-docs 100 \
--num_threads 16 \
--lambda_min 1.0 \
[--lambda_max 10.0] \
[--lambda_step 1.0] \
[--wandb_project WANDB_PROJECT] \
[--wandb_name WANDB_NAME] \
[--wandb_name ] 
```

## Pretraining

To reproduce the pretraining of Spider, run:

```bash
python train_dense_encoder.py \
--pretraining \
--encoder_model_type hf_bert \
--pretrained_model_cfg bert-base-uncased \
--weight_sharing \
--do_lower_case \
--train_file "PRETRAINING_DATA_DIR/recurring_*.pkl" \
--tokenized_passages "PRETRAINING_DATA_DIR/tokenized_*.pkl" \
--output_dir PRETRAINING_DIR \
--query_transformation random \
--keep_answer_prob 0.5 \
--batch_size 1024 \
--update_steps 200000 \
--sequence_length 240 \
--question_sequence_length 64 \
--learning_rate 2e-5 \
--warmup_steps 2000 \
--max_grad_norm 2.0 \
--seed 12345 \
--eval_steps -1 \
--log_batch_step 10000000 \
--train_rolling_loss_step 100 \
--wandb_project $WANDB_PROJECT \
--wandb_name $WANDB_RUN_NAME \
--fp16
```

## Fine-Tuning

To run fine-tuning (for example on Natural Questions or TriviaQA), you'll first need to download train and dev files from DPR repo.

```bash
python download_data.py --resource data.retriever.nq-train 
python download_data.py --resource data.retriever.nq-dev
python download_data.py --resource data.retriever.trivia-train 
python download_data.py --resource data.retriever.trivia-dev
```
The files will be downloaded to `./downloads/data/retriever/{nq|trivia}-{train|dev}.json`.

See [here](https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py) the list of all available resources.   
Alternatively, if you have your own training data, make sure it adheres to the same format.

To fine-tune your model (or Spider), run:
```bash
python train_dense_encoder.py \
--max_grad_norm 2.0 \
--encoder_model_type hf_bert \
--pretrained_model_cfg tau/spider \
[--model_file MODEL_CKPT_FILE] \
--load_only_model \
--do_lower_case \
--seed 12345 \
--sequence_length 240 \
--warmup_steps 1000 \
--batch_size 128 \
--train_file TRAIN_JSON \
--dev_file DEV_JSON \
--output_dir OUTPUT_DIR \
--fp16 \
--learning_rate 1e-05 \
--num_train_epochs 40 \
--dev_batch_size 128 \
--val_av_rank_start_epoch 39 \
--log_batch_step 1000000 \
--train_rolling_loss_step 10 \
[--no_wandb] \
[--wandb_project WANDB_PROJECT] \
[--wandb_name WANDB_NAME] 
```
Note that `--model_file` is used for checkpoint files saved in `train_dense_encoder.py`, 
so use it only for your own pretrained models.  
Also, you can replace `tau/spider` with `bert-base-uncased` in order to reproduce original DPR training.
`

## Convert Model Checkpoints to Hugging Face and Upload to Hub

You can convert your trained model checkpoints to Hugging Face format and automatically upload them to the hub:
```bash
python convert_checkpoint_to_hf.py \
--ckpt_path CKPT_PATH \
--output_dir OUTPUT_DIR \
--model_type ["shared", "question", "context"] \
[--hf_model_name HF_USER/HF_MODEL_NAME]
```

## Citation

If you find our code or models helpful, please cite our paper:
```
@inproceedings{ram-etal-2022-learning,
    title = "Learning to Retrieve Passages without Supervision",
    author = "Ram, Ori  and
      Shachaf, Gal  and
      Levy, Omer" and
      Berant, Jonathan  and
      Globerson, Amir",
    booktitle = "NAACL 2022",
    year = "2022",
}
```